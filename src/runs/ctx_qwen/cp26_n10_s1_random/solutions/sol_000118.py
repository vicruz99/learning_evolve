# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000066 (state 7dd8b726) state=b8add980 sum of radii=2.624511 correctness=1.0
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
    """Objective for joint optimization: minimize negative sum of radii."""
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
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)  # Self-distance placeholder
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(d2[mask] - r_sum_sq[mask])
    return np.concatenate(cons)

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    
    c_obj = np.ones(n) * -1.0
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)  # Avoid self-constraints
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and np.isfinite(res.fun):
        return res.x, -res.fun
    return np.full(n, 1e-6), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    
    # 1. Generate diverse initial configurations
    configs = []
    row_dist = [5, 6, 5, 6, 4]  # Optimal row distribution for N=26 hex packing
    
    for r_init in [0.09, 0.10, 0.10135, 0.11]:
        cfg = get_hex_config(n, row_dist, r_init)
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # Rotated hex grids to break symmetry
    for angle in [0.05, -0.05, 0.1, -0.1, 0.15]:
        cfg = get_hex_config(n, row_dist, 0.10)
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        cfg = cfg @ rot
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0))
        cfg = cfg * 0.88 + 0.06
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # Random dense starts
    np.random.seed(42)
    for _ in range(4):
        configs.append(np.random.uniform(0.15, 0.85, (n, 2)))
        
    # 2. Joint optimization phase
    for cfg in configs:
        r0 = np.full(n, 0.09)
        x0 = np.concatenate([cfg.flatten(), r0])
        
        try:
            res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                          bounds=bounds_vars,
                          constraints={'type': 'ineq', 'fun': joint_constraints, 'args': (n,)},
                          options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                cons_val = joint_constraints(res.x, n)
                if np.min(cons_val) > -1e-5:
                    c_opt = res.x[:2*n].reshape(n, 2)
                    # LP refinement guarantees optimal radii for these centers
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            pass
            
    # 3. Perturbation refinement to escape local minima
    if best_centers is not None:
        for _ in range(12):
            pert = best_centers + np.random.uniform(-0.004, 0.004, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r0 = np.full(n, 0.09)
            x0 = np.concatenate([pert.flatten(), r0])
            try:
                res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                              bounds=bounds_vars,
                              constraints={'type': 'ineq', 'fun': joint_constraints, 'args': (n,)},
                              options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun):
                    cons_val = joint_constraints(res.x, n)
                    if np.min(cons_val) > -1e-5:
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
        best_centers = np.clip(get_hex_config(n, row_dist, 0.09), 0.1, 0.9)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # 4. Final safety scaling to strictly satisfy 1e-12 validator tolerance
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
                
    # Apply with minimal margin
    r *= scale * 0.9999995
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum
