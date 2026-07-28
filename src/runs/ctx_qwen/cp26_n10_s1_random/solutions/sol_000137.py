# sol_000137 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000067 (state 3fcdd2a7) state=33344479 sum of radii=1.234598 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_objective(vars_array, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_array[2*n:3*n])

def compute_constraints(vars_array, n, triu_idx):
    """Returns array of constraint values >= 0 for valid packing"""
    x = vars_array[0:n]
    y = vars_array[n:2*n]
    r = vars_array[2*n:3*n]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    c_pairwise = dist_sq[triu_idx] - r_sum[triu_idx]**2
    
    return np.concatenate([c_boundary, c_pairwise])

def generate_initial_configs(n, seed=42):
    """Generates diverse high-quality initial configurations"""
    rng = np.random.default_rng(seed)
    configs = []
    r0 = 0.101
    
    # Pattern 1: Hexagonal layout 5, 6, 5, 6, 4
    rows = [5, 6, 5, 6, 4]
    pts = []
    y = r0
    for i, cnt in enumerate(rows):
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3)
    configs.append(np.array(pts[:n]))
    
    # Pattern 2: Hexagonal layout 6, 5, 6, 5, 4
    rows2 = [6, 5, 6, 5, 4]
    pts2 = []
    y = r0
    for i, cnt in enumerate(rows2):
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            pts2.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3)
    configs.append(np.array(pts2[:n]))
    
    # Perturbations of hex patterns
    for cfg in [configs[0], configs[1]]:
        for _ in range(6):
            pert = cfg + rng.uniform(-0.025, 0.025, cfg.shape)
            configs.append(np.clip(pert, 0.05, 0.95))
            
    # Random dense starts
    for _ in range(5):
        rc = np.clip(rng.uniform(0.08, 0.92, (n, 2)), 0.05, 0.95)
        configs.append(rc)
        
    return configs

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-6, 0.5)] * n
    
    best_sum = -np.inf
    best_vars = None
    
    configs = generate_initial_configs(n)
    
    # Phase 1: Multi-start constrained optimization
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), np.full(n, 0.10)])
        
        try:
            res = minimize(
                compute_objective,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n, triu_idx)},
                options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_vals = compute_constraints(res.x, n, triu_idx)
                if np.min(c_vals) >= -1e-5:
                    s = np.sum(res.x[2*n:3*n])
                    if s > best_sum:
                        best_sum = s
                        best_vars = res.x.copy()
        except Exception:
            continue

    if best_vars is None:
        best_vars = np.concatenate([configs[0].flatten(), np.full(n, 0.095)])
        
    # Phase 2: Local perturbation search to escape local minima
    rng = np.random.default_rng(123)
    for _ in range(10):
        pert_scale = 0.005
        pert = rng.uniform(-pert_scale, pert_scale, 2*n)
        x_pert = best_vars.copy()
        x_pert[:2*n] = np.clip(best_vars[:2*n] + pert, 0.05, 0.95)
        
        try:
            res = minimize(
                compute_objective,
                x_pert,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n, triu_idx)},
                options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_vals = compute_constraints(res.x, n, triu_idx)
                if np.min(c_vals) >= -1e-5:
                    s = np.sum(res.x[2*n:3*n])
                    if s > best_sum:
                        best_sum = s
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    centers = best_vars[:2*n].reshape(n, 2)
    radii_init = best_vars[2*n:3*n]
    
    # Phase 3: LP refinement to maximize radii for fixed optimal centers
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    
    c_obj = np.ones(n) * -1.0
    A_ub = []
    b_ub = []
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    bounds_lp = [(0.0, lim) for lim in limits]
    
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res_lp.success and np.isfinite(res_lp.fun):
            final_radii = res_lp.x * 0.99999
            return centers, final_radii, float(np.sum(final_radii))
    except Exception:
        pass
        
    # Fallback scaling if LP fails
    scale = 1.0
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii_init[i]
        if r > 1e-9:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            rs = radii_init[i] + radii_init[j]
            if rs > 1e-9:
                scale = min(scale, d/rs)
                
    final_radii = radii_init * scale * 0.99995
    return centers, final_radii, float(np.sum(final_radii))
