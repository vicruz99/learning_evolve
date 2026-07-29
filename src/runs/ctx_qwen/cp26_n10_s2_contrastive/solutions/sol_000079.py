# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000051 (state 921aef56) state=934d3c1d sum of radii=2.622092 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
N_PAIRS = len(I)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Boundary and non-overlap constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Pairwise distance constraints
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    dists = np.hypot(dx, dy)
    c_overlap = dists - (r[I] + r[J])
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_overlap, c_bound])

def solve_lp_radii(centers):
    """Given fixed centers, find radii that maximize sum via LP."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((N_PAIRS, n))
    b_ub = np.zeros(N_PAIRS)
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub[:] = np.hypot(dx, dy)
    
    A_ub[np.arange(N_PAIRS), I] = 1.0
    A_ub[np.arange(N_PAIRS), J] = 1.0
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='interior-point')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.zeros(n), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = []
    rng_global = np.random.RandomState(42)
    
    # 1. Hexagonal lattice initializations with varying parameters
    for seed in range(30):
        rng = np.random.RandomState(seed * 17 + 1)
        centers = np.zeros((N, 2))
        idx = 0
        row = 0
        s = 0.13 + rng.uniform(0, 0.06)
        y = s / 2
        while idx < N and y < 1.0 - s / 2:
            x_start = s / 2 + (row % 2) * s / 2
            col = 0
            while x_start + col * s < 1.0 - s / 2 and idx < N:
                centers[idx, 0] = x_start + col * s
                centers[idx, 1] = y
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
            
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
            
        centers += rng.normal(0, 0.025, centers.shape)
        centers = np.clip(centers, 0.04, 0.96)
        
        r_lp, _ = solve_lp_radii(centers)
        x0 = np.zeros(3 * N)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = np.maximum(r_lp * 0.92, 1e-5)
        inits.append(x0)
        
    # 2. Random feasible initializations
    for seed in range(40):
        rng = np.random.RandomState(seed * 31 + 2)
        centers = rng.uniform(0.15, 0.85, (N, 2))
        r_lp, _ = solve_lp_radii(centers)
        x0 = np.zeros(3 * N)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = np.maximum(r_lp * 0.90, 1e-5)
        inits.append(x0)
        
    # Phase 1: Broad multi-start search
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                c_opt = np.column_stack((cx, cy))
                r_opt, s_opt = solve_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt
                    best_radii = r_opt
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement around the best solution
    if best_centers is not None:
        rng = np.random.RandomState(123)
        for step in range(35):
            # Scale noise to explore different basin depths
            noise_scale = 0.008 * (0.85 ** (step // 10))
            c_pert = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, _ = solve_lp_radii(c_pert)
            
            x_pert = np.zeros(3 * N)
            x_pert[0::3] = c_pert[:, 0]
            x_pert[1::3] = c_pert[:, 1]
            x_pert[2::3] = np.maximum(r_pert * 0.95, 1e-5)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    c_opt = np.column_stack((cx, cy))
                    r_opt, s_opt = solve_lp_radii(c_opt)
                    
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = c_opt
                        best_radii = r_opt
            except Exception:
                continue
                
    # Fallback safety net
    if best_centers is None:
        centers = np.random.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = solve_lp_radii(centers)
        best_centers = centers
        
    # Phase 3: Strict post-processing to guarantee validity
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        x, y = best_centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r - 1e-9)
            
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                             best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
