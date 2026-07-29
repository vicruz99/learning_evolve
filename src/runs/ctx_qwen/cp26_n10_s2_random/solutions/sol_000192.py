# sol_000192 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000163 (state a7643fac) state=f8f3bd9d sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute constant structure of the LP constraint matrix
pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(pair_indices)

A_ub_structure = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(pair_indices):
    A_ub_structure[k, i] = 1.0
    A_ub_structure[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_ub_structure[base, i] = 1.0
    A_ub_structure[base + 1, i] = 1.0
    A_ub_structure[base + 2, i] = 1.0
    A_ub_structure[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes gradient via duals."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    dx = c[:, 0, np.newaxis] - c[np.newaxis, :, 0]
    dy = c[:, 1, np.newaxis] - c[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    b_ub = np.empty(N_PAIRS + 4 * N)
    for k, (i, j) in enumerate(pair_indices):
        b_ub[k] = dists[i, j]
    for i in range(N):
        base = N_PAIRS + 4 * i
        b_ub[base] = c[i, 0]
        b_ub[base + 1] = 1.0 - c[i, 0]
        b_ub[base + 2] = c[i, 1]
        b_ub[base + 3] = 1.0 - c[i, 1]
        
    bounds_r = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_ub_structure, b_ub=b_ub, 
                  bounds=bounds_r, method='highs')
    
    if not res.success:
        return np.full(N, 0.0), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(N_PAIRS + 4 * N)
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(c)
    
    for k, (i, j) in enumerate(pair_indices):
        mu = duals[k]
        if mu > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
                
    for i in range(N):
        base = N_PAIRS + 4 * i
        grad[i, 0] += duals[base] - duals[base + 1]
        grad[i, 1] += duals[base + 2] - duals[base + 3]
        
    return radii, s_sum, grad

def obj_centers(v):
    """Objective for center optimization: minimize negative sum of radii."""
    _, s, _ = solve_lp_and_grad(v.reshape(N, 2))
    return -s

def jac_centers(v):
    """Jacobian for center optimization."""
    _, _, g = solve_lp_and_grad(v.reshape(N, 2))
    return -g.flatten()

def optimize_centers_lbfgs(c0, max_iter=800):
    """Optimizes centers using L-BFGS-B with LP-derived gradient."""
    bounds = [(0.002, 0.998)] * (2 * N)
    v0 = c0.flatten()
    try:
        res = minimize(obj_centers, v0, method='L-BFGS-B', jac=jac_centers, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
        return c_opt, r_opt, s_opt
    except Exception:
        return c0, np.full(N, 0.05), 0.0

def generate_diverse_starts(n, rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [6, 6, 5, 5, 4], [5, 7, 5, 6], [7, 6, 6, 7]
    ]
    
    for pat in patterns:
        for r_est in [0.085, 0.095, 0.105, 0.115]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < n:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:n])
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Random dense starts
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Corner/edge heavy starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (n, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c[4:8] = [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
        starts.append(c)
        
    return starts

def obj_joint(v):
    return -np.sum(v[2*N:])

def cons_joint(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
    idx_i, idx_j = np.triu_indices(N, 1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    dr = r[idx_i] + r[idx_j]
    cons.append(np.sqrt(dx**2 + dy**2) - dr)
    return np.concatenate(cons)

def repair(centers, radii):
    """Deterministic repair to guarantee validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_diverse_starts(N, rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c0 in starts:
        c_opt, r_opt, s_opt = optimize_centers_lbfgs(c0, max_iter=600)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Iterative Perturbation & Local Search
    for _ in range(120):
        c_pert = best_c.copy()
        # Jitter a random subset of circles
        subset_size = rng.integers(2, 6)
        idx = rng.choice(N, subset_size, replace=False)
        c_pert[idx] += rng.normal(0, 0.005, (subset_size, 2))
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        c_opt, r_opt, s_opt = optimize_centers_lbfgs(c_pert, max_iter=400)
        if s_opt > best_sum + 1e-8:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    # Phase 3: Joint SLSQP Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 8000, 'ftol': 1e-14})
        if np.min(cons_joint(res.x)) >= -1e-9:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_sum = s
                best_c = res.x[:2*N].reshape(N, 2).copy()
                best_r = res.x[2*N:].copy()
    except Exception:
        pass
        
    # Phase 4: Final L-BFGS-B polish on best configuration
    c_final, r_final, s_final = optimize_centers_lbfgs(best_c, max_iter=1000)
    if s_final > best_sum:
        best_sum = s_final
        best_c = c_final.copy()
        best_r = r_final.copy()
        
    # Phase 5: Strict Numerical Repair
    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
