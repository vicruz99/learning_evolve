# sol_000223 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000216 (state 64a1292d) state=d863e954 sum of radii=2.516691 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N_CIRCLES = 26
TRIU_I, TRIU_J = np.triu_indices(N_CIRCLES, k=1)
A_UB_FIXED = np.zeros((len(TRIU_I), N_CIRCLES))
for k, (i, j) in enumerate(zip(TRIU_I, TRIU_J)):
    A_UB_FIXED[k, i] = 1.0
    A_UB_FIXED[k, j] = 1.0

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(lim, 1e-10)) for lim in limits]
    
    diffs = centers[TRIU_I] - centers[TRIU_J]
    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
    
    c_obj = -np.ones(n)
    try:
        res = linprog(c_obj, A_ub=A_UB_FIXED, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def obj_slqp(x):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(x[2*N_CIRCLES:])

def cons_slqp(x):
    """Inequality constraints >= 0 for joint optimization."""
    c = x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = x[2*N_CIRCLES:]
    # Boundary constraints
    con = np.concatenate([c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r])
    # Pairwise non-overlap constraints
    dx = c[:,0,None] - c[None,:,0]
    dy = c[:,1,None] - c[None,:,1]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_sum = r[:,None] + r[None,:]
    con = np.concatenate([con, dists[TRIU_I, TRIU_J] - r_sum[TRIU_I, TRIU_J]])
    return con

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    configs = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [5,4,6,6,5], [5,6,4,6,5], [6,5,5,6,4],
        [5,7,5,5,4], [5,5,5,5,6], [7,5,5,5,4], [4,5,6,5,6]
    ]
    
    for pat in patterns:
        if sum(pat) != n: continue
        r0 = 0.095
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        configs.append(np.array(pts))
        
    for _ in range(15):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Simulated Annealing on centers with LP evaluation
    initial_step = 0.04
    temp_start = 0.02
    iters_per_cfg = 5000
    
    for cfg in configs:
        centers = cfg.copy()
        centers = np.clip(centers, 0.05, 0.95)
        
        radii, current_sum = solve_radii_lp(centers)
        if current_sum <= best_sum: continue
        
        step = initial_step
        temp = temp_start
        
        for it in range(iters_per_cfg):
            i = rng.integers(n)
            old_c = centers[i].copy()
            
            centers[i] += rng.uniform(-step, step, 2)
            centers[i] = np.clip(centers[i], 0.02, 0.98)
            
            new_radii, new_sum = solve_radii_lp(centers)
            
            delta = new_sum - current_sum
            if delta > 0 or rng.random() < np.exp(np.clip(delta / max(temp, 1e-9), -50, 50)):
                current_sum = new_sum
                radii = new_radii
                step *= 0.9995
                temp *= 0.9994
            else:
                centers[i] = old_c
                
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Phase 2: SLSQP Joint Polish
    if best_centers is not None:
        x0 = np.zeros(3 * n)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_radii * 0.995
        
        bounds_slqp = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * n
        
        try:
            res = minimize(obj_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints={'type': 'ineq', 'fun': cons_slqp},
                           options={'maxiter': 8000, 'ftol': 1e-13})
            if res.success and np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt, s_opt = solve_radii_lp(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt
                    best_radii = r_opt
        except Exception:
            pass
            
    # Phase 3: Strict Safety Scaling
    if best_centers is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i,0], best_centers[i,1], best_radii[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
                
        dists = np.sqrt(np.sum((best_centers[:,None,:] - best_centers[None,:,:])**2, axis=2))
        for i in range(n):
            for j in range(i+1, n):
                rs = best_radii[i] + best_radii[j]
                if rs > 1e-12:
                    scale = min(scale, dists[i,j] / rs)
                    
        best_radii *= scale * 0.9999995
        best_sum = float(np.sum(best_radii))
    else:
        best_centers = rng.uniform(0.1, 0.9, (n, 2))
        best_radii = np.full(n, 0.05)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
