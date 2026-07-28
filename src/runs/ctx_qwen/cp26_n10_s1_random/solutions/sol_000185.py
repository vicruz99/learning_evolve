# sol_000185 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=23ba6f3c sum of radii=2.624553 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_hex_config(n, row_counts, r_init):
    """Generates a hexagonal lattice initialization with specified row counts."""
    pts = []
    y = r_init
    row_idx = 0
    for cnt in row_counts:
        shift = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(cnt):
            if len(pts) < n:
                pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
    return np.array(pts[:n])

def joint_objective(vars_array, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_array[2*n:])

def joint_constraints(vars_array, n):
    """Returns inequality constraints >= 0 for valid packing."""
    c = vars_array[:2*n].reshape(n, 2)
    r = vars_array[2*n:]
    cons = []
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    d2 = dx**2 + dy**2
    r_sum_sq = (r[idx_i] + r[idx_j])**2
    cons.append(d2 - r_sum_sq)
    
    return np.concatenate(cons)

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    
    c_obj = np.ones(n) * -1.0
    bounds = [(0.0, lim) for lim in limits]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    b_ub = dists.copy()
    
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': joint_constraints, 'args': (n,)}
    
    rng = np.random.default_rng(42)
    configs = []
    
    # Diverse hexagonal row distributions known to be near-optimal for N=26
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5], [6, 5, 5, 6, 4]
    ]
    
    for dist in row_dists:
        if sum(dist) < n: continue
        cfg = get_hex_config(n, dist, 0.10)
        configs.append(cfg)
        
        # Add perturbations to break symmetry
        for _ in range(2):
            cfg_p = cfg + rng.uniform(-0.02, 0.02, cfg.shape)
            configs.append(np.clip(cfg_p, 0.05, 0.95))
            
        # Add rotations to explore different alignments with the square
        for angle in [0.03, -0.03, 0.07, -0.07]:
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            cfg_r = cfg @ rot
            cfg_r = cfg_r - cfg_r.min(axis=0)
            cfg_r = cfg_r / cfg_r.max(axis=0) * 0.85 + 0.075
            configs.append(cfg_r)

    # Add purely random dense starts
    for _ in range(4):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Multi-start Joint Optimization
    for cfg in configs:
        r0 = np.full(n, 0.095)
        x0 = np.concatenate([cfg.flatten(), r0])
        
        try:
            res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                          bounds=bounds_vars, constraints=cons_dict,
                          options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                cons_val = joint_constraints(res.x, n)
                if np.min(cons_val) > -1e-6:
                    c_opt = res.x[:2*n].reshape(n, 2)
                    # LP refinement guarantees optimal radii for these centers
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement around the best configuration found
    if best_centers is not None:
        for _ in range(12):
            pert = best_centers + rng.uniform(-0.004, 0.004, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r0 = np.full(n, 0.095)
            x0 = np.concatenate([pert.flatten(), r0])
            try:
                res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                              bounds=bounds_vars, constraints=cons_dict,
                              options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun):
                    cons_val = joint_constraints(res.x, n)
                    if np.min(cons_val) > -1e-6:
                        c_opt = res.x[:2*n].reshape(n, 2)
                        r_lp, s_lp = solve_radii_lp(c_opt)
                        if s_lp > best_sum:
                            best_sum = s_lp
                            best_centers = c_opt.copy()
                            best_radii = r_lp.copy()
            except Exception:
                pass
                
    # Fallback safety net
    if best_centers is None:
        best_centers = np.clip(get_hex_config(n, [5, 6, 5, 6, 4], 0.09), 0.1, 0.9)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Phase 3: Strict safety scaling to guarantee validity within 1e-12 tolerance
    scale = 1.0
    c = best_centers
    r = best_radii
    for i in range(n):
        if r[i] > 1e-12:
            scale = min(scale, c[i,0]/r[i], (1-c[i,0])/r[i], c[i,1]/r[i], (1-c[i,1])/r[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i]-c[j])
            r_sum = r[i] + r[j]
            if r_sum > 1e-12:
                scale = min(scale, d / r_sum)
                
    # Apply with minimal margin to preserve maximum area
    r *= scale * 0.9999999
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum
