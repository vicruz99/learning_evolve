# sol_000194 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000163 (state a7643fac) state=fadd3c64 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)

# Precompute constant structure of the LP constraint matrix
A_ub_struct = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_ub_struct[k, i] = 1.0
    A_ub_struct[k, j] = 1.0
for i in range(N):
    b = N_PAIRS + 4 * i
    A_ub_struct[b, i] = 1.0
    A_ub_struct[b+1, i] = 1.0
    A_ub_struct[b+2, i] = 1.0
    A_ub_struct[b+3, i] = 1.0

def solve_lp(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(N_PAIRS + 4 * N)
    for k, (i, j) in enumerate(PAIR_INDICES):
        b_ub[k] = dists[i, j]
    for i in range(N):
        b = N_PAIRS + 4 * i
        b_ub[b] = c[i, 0]
        b_ub[b+1] = 1.0 - c[i, 0]
        b_ub[b+2] = c[i, 1]
        b_ub[b+3] = 1.0 - c[i, 1]
        
    res = linprog(-np.ones(N), A_ub=A_ub_struct, b_ub=b_ub,
                  bounds=[(0.0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals safely across scipy versions
    duals = np.zeros_like(b_ub)
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(c)
    # Pairwise repulsion forces from active distance constraints
    for k, (i, j) in enumerate(PAIR_INDICES):
        mu = duals[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                v = (c[i] - c[j]) / d
                grad[i] += mu * v
                grad[j] -= mu * v
                
    # Boundary forces from active wall constraints
    for i in range(N):
        b = N_PAIRS + 4 * i
        grad[i, 0] += duals[b] - duals[b+1]
        grad[i, 1] += duals[b+2] - duals[b+3]
        
    return radii, s_sum, grad

def objective_lp(x):
    """Objective for center optimization: minimize negative sum of radii."""
    return -solve_lp(x.reshape(N, 2))[1]

def optimize_gradient(c0, max_iter=2500, rng=None):
    """Runs gradient ascent on centers to maximize sum of radii."""
    c = c0.copy()
    best_c, best_s = c.copy(), -1.0
    step = 0.006
    for k in range(max_iter):
        _, s, g = solve_lp(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        c = c + step * (g / gn)
        c = np.clip(c, 0.01, 0.99)
        
        # Adaptive step decay
        if k % 200 == 0 and k > 0:
            step *= 0.94
            
        # Periodic jitter to escape plateaus
        if k % 400 == 0 and rng is not None:
            c += rng.normal(0, 0.0006, c.shape)
            c = np.clip(c, 0.02, 0.98)
            
    return best_c, best_s

def generate_starts(n, rng):
    """Generates a wide variety of initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.088, 0.095, 0.102, 0.109]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:n])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Random dense starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Corner-heavy starts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (n, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_joint(v):
    """Computes boundary and non-overlap constraints (must be >= 0)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dr = r[idx_i] + r[idx_j]
    cons.append(np.sqrt(dx**2 + dy**2) - dr)
    return np.concatenate(cons)

def objective_single(centers, idx, p):
    """Objective for single-circle optimization."""
    c_tmp = centers.copy()
    c_tmp[idx] = p
    return -solve_lp(c_tmp)[1]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(N, rng)
    
    # --- Phase 1: Multi-start Gradient Ascent ---
    for c0 in starts:
        c_opt, s_opt = optimize_gradient(c0, max_iter=3000, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp(best_c)
    else:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp(best_c)
        
    # --- Phase 2: Powell Derivative-Free Refinement ---
    bounds_c = [(0.0, 1.0)] * (2 * N)
    for _ in range(3):
        try:
            res = minimize(objective_lp, best_c.flatten(), method='Powell',
                          bounds=bounds_c, options={'maxiter': 1000, 'ftol': 1e-12})
            c_pow = res.x.reshape(N, 2)
            r_pow, s_pow, _ = solve_lp(c_pow)
            if s_pow > best_sum:
                best_sum = s_pow
                best_c = c_pow.copy()
                best_r = r_pow.copy()
        except Exception:
            pass
            
    # --- Phase 3: Single-Circle Coordinate Descent ---
    for _ in range(4):
        improved = True
        while improved:
            improved = False
            for i in range(N):
                try:
                    res = minimize(objective_single, best_c[i], args=(best_c, i), 
                                  method='Nelder-Mead', options={'maxiter': 400})
                    if -res.fun > best_sum:
                        best_sum = -res.fun
                        best_c[i] = res.x
                        _, best_r, _ = solve_lp(best_c)
                        improved = True
                except Exception:
                    pass
                    
    # --- Phase 4: SLSQP Joint Polish ---
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    for _ in range(2):
        v_pert = v0 + rng.normal(0, 0.0012, v0.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.4)
        try:
            res = minimize(objective_joint, v_pert, method='SLSQP', bounds=bounds_j,
                          constraints={'type': 'ineq', 'fun': constraints_joint},
                          options={'maxiter': 6000, 'ftol': 1e-13})
            if np.min(constraints_joint(res.x)) >= -1e-9:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # --- Phase 5: Strict Numerical Repair ---
    centers = best_c.copy()
    radii = best_r.copy()
    for _ in range(150):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
