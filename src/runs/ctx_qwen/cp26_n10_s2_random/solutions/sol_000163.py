# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000161 (state 3495d006) state=a7643fac sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PAIR_INDICES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_PAIRS = len(PAIR_INDICES)

# Precompute constant structure of the LP constraint matrix
A_ub_structure = np.zeros((N_PAIRS + 4 * N, N))
for k, (i, j) in enumerate(PAIR_INDICES):
    A_ub_structure[k, i] = 1.0
    A_ub_structure[k, j] = 1.0
for i in range(N):
    base = N_PAIRS + 4 * i
    A_ub_structure[base, i] = 1.0
    A_ub_structure[base + 1, i] = 1.0
    A_ub_structure[base + 2, i] = 1.0
    A_ub_structure[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
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
        base = N_PAIRS + 4 * i
        b_ub[base] = c[i, 0]
        b_ub[base + 1] = 1.0 - c[i, 0]
        b_ub[base + 2] = c[i, 1]
        b_ub[base + 3] = 1.0 - c[i, 1]
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_ub_structure, b_ub=b_ub, 
                  bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals safely across scipy versions
    duals = np.zeros(b_ub.shape[0])
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
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
                
    # Boundary forces from active wall constraints
    for i in range(N):
        base = N_PAIRS + 4 * i
        mu_L = duals[base]
        mu_R = duals[base + 1]
        mu_B = duals[base + 2]
        mu_T = duals[base + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, s_sum, grad

def optimize_gradient(centers0, steps=2500, init_step=0.008):
    """Runs gradient ascent on centers to maximize sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    no_improve = 0
    
    for k in range(steps):
        radii, curr_sum, grad = solve_lp_and_grad(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        # Adaptive step decay
        if no_improve > 70:
            step *= 0.6
        elif no_improve > 30:
            step *= 0.85
            
        if step < 1e-10:
            break
            
        centers = centers + step * grad
        centers = np.clip(centers, 0.01, 0.99)
            
        # Periodic jitter to escape plateaus
        if k % 250 == 0 and k > 0:
            centers += np.random.normal(0, 0.0005, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            
    return best_centers, best_sum

def generate_diverse_starts(n, rng):
    """Generates a wide variety of initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.088, 0.094, 0.100, 0.106]:
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
    for _ in range(12):
        starts.append(rng.uniform(0.12, 0.88, (n, 2)))
        
    return starts

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_diverse_starts(N, rng)
    
    # --- Phase 1: Gradient ascent from diverse starts ---
    for c0 in starts:
        c_opt, s_opt = optimize_gradient(c0, steps=2000, init_step=0.012)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp_and_grad(best_c)
    else:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # --- Phase 2: Iterative Perturbation & Restart ---
    for _ in range(6):
        c_pert = best_c + rng.normal(0, 0.007, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        c_opt, s_opt = optimize_gradient(c_pert, steps=1500, init_step=0.008)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # --- Phase 3: SLSQP Joint Polish ---
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    for _ in range(4):
        v_pert = v0 + rng.normal(0, 0.0015, v0.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.4)
        try:
            res = minimize(obj_joint, v_pert, method='SLSQP', bounds=bounds_j,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 6000, 'ftol': 1e-13})
            if np.min(cons_joint(res.x)) >= -1e-9:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # --- Phase 4: Simulated Annealing on Centers with LP Radii ---
    c_curr = best_c.copy()
    s_curr = best_sum
    r_curr = best_r.copy()
    T = 0.004
    for step in range(600):
        T *= 0.997
        c_new = c_curr + rng.normal(0, T, c_curr.shape)
        c_new = np.clip(c_new, 0.02, 0.98)
        r_new, s_new, _ = solve_lp_and_grad(c_new)
        
        if s_new > s_curr:
            c_curr = c_new
            s_curr = s_new
            r_curr = r_new
            if s_curr > best_sum:
                best_sum = s_curr
                best_c = c_curr.copy()
                best_r = r_curr.copy()
        else:
            if rng.random() < np.exp((s_new - s_curr) / max(T * 3.0, 1e-7)):
                c_curr = c_new
                s_curr = s_new
                r_curr = r_new
                
    # --- Phase 5: Final SLSQP Polish ---
    r_lp, _, _ = solve_lp_and_grad(best_c)
    v0_final = np.concatenate([best_c.flatten(), r_lp])
    try:
        res = minimize(obj_joint, v0_final, method='SLSQP', bounds=bounds_j,
                      constraints={'type': 'ineq', 'fun': cons_joint},
                      options={'maxiter': 5000, 'ftol': 1e-13})
        if np.min(cons_joint(res.x)) >= -1e-9:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_sum = s
                best_c = res.x[:2*N].reshape(N, 2).copy()
                best_r = res.x[2*N:].copy()
    except Exception:
        pass
        
    # --- Phase 6: Strict Numerical Repair ---
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
