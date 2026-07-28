# sol_000240 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000177 (state 0ce77dda) state=498328fc sum of radii=2.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def get_lp_data(centers, n):
    """Constructs the LP matrix and vector to maximize sum of radii."""
    c_obj = -np.ones(n)
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((4 * n + num_pairs, n))
    b_ub = np.zeros(4 * n + num_pairs)
    
    idx = 0
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        # r_i <= 1 - x
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        # r_i <= y
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        # r_i <= 1 - y
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    bounds = [(0, None)] * n
    return c_obj, A_ub, b_ub, bounds

def eval_objective_and_grad(centers_flat, n, pair_i, pair_j):
    """Evaluates the max sum of radii and its gradient w.r.t centers using LP duals."""
    centers = centers_flat.reshape(n, 2)
    c_obj, A_ub, b_ub, bounds = get_lp_data(centers, n)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return -1e9, np.zeros_like(centers_flat)
        
    sum_r = -res.fun
    duals = np.asarray(res.ineqlin.marginals).ravel()
    
    grad = np.zeros((n, 2))
    
    # Boundary gradient contributions
    for i in range(n):
        grad[i, 0] = duals[4*i] - duals[4*i+1]
        grad[i, 1] = duals[4*i+2] - duals[4*i+3]
        
    # Pairwise gradient contributions
    offset = 4 * n
    for k in range(len(pair_i)):
        lam = duals[offset + k]
        if lam < 1e-12:
            continue
        i, j = pair_i[k], pair_j[k]
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        d = np.hypot(dx, dy)
        if d < 1e-12:
            continue
        factor = lam / d
        grad[i, 0] += factor * dx
        grad[i, 1] += factor * dy
        grad[j, 0] -= factor * dx
        grad[j, 1] -= factor * dy
        
    return sum_r, grad.flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    pair_i, pair_j = np.triu_indices(n, k=1)
    bounds_centers = [(0.005, 0.995)] * (2 * n)
    
    def objective_fn(x):
        val, _ = eval_objective_and_grad(x, n, pair_i, pair_j)
        return -val
        
    def grad_fn(x):
        _, g = eval_objective_and_grad(x, n, pair_i, pair_j)
        return -g
        
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5]
    ]
    
    for p in patterns:
        pts = []
        r0 = 0.09
        y = r0
        for ri, cnt in enumerate(p):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
        while len(pts) < n:
            pts.append([0.5, 0.5])
        pts = np.array(pts[:n])
        inits.append(pts)
        
        # Perturbed variants
        for _ in range(3):
            p_pert = np.clip(pts + rng.uniform(-0.025, 0.025, pts.shape), 0.05, 0.95)
            inits.append(p_pert)
            
    # Random starts
    for _ in range(12):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Optimize each start
    for cfg in inits:
        x0 = cfg.flatten()
        try:
            res = minimize(objective_fn, x0, jac=grad_fn, method='L-BFGS-B', 
                           bounds=bounds_centers, options={'maxiter': 1500, 'ftol': 1e-16})
            
            curr_centers = res.x.reshape(n, 2)
            
            # Final LP to get exact radii for optimized centers
            c_obj, A_ub, b_ub, bounds_r = get_lp_data(curr_centers, n)
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
            if res_lp.success:
                radii = res_lp.x
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = curr_centers.copy()
                    best_radii = radii.copy()
        except Exception:
            continue
            
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        c_obj, A_ub, b_ub, bounds_r = get_lp_data(best_centers, n)
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        best_radii = res_lp.x
        best_sum = np.sum(best_radii)
        
    # Local jitter & refinement to escape flat regions
    for _ in range(20):
        jitter = np.clip(best_centers + rng.uniform(-0.008, 0.008, best_centers.shape), 0.01, 0.99)
        x0 = jitter.flatten()
        try:
            res = minimize(objective_fn, x0, jac=grad_fn, method='L-BFGS-B', 
                           bounds=bounds_centers, options={'maxiter': 800, 'ftol': 1e-16})
            jitter_centers = res.x.reshape(n, 2)
            c_obj, A_ub, b_ub, bounds_r = get_lp_data(jitter_centers, n)
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
            if res_lp.success:
                s = np.sum(res_lp.x)
                if s > best_sum:
                    best_sum = s
                    best_centers = jitter_centers.copy()
                    best_radii = res_lp.x.copy()
        except Exception:
            pass

    # Safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
