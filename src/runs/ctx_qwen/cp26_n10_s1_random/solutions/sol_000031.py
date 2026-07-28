# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state bde5dee5) state=ffbce667 sum of radii=2.615345 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Precompute indices for pairwise constraints to avoid recomputation
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    def objective(vars_array):
        # Maximize sum of radii => minimize negative sum
        radii = vars_array[2*n:3*n]
        return -np.sum(radii)
        
    def constraint_func(vars_array):
        centers = vars_array[:2*n].reshape(n, 2)
        radii = vars_array[2*n:3*n]
        
        # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        b_cons = np.concatenate([
            centers[:, 0] - radii,
            1.0 - centers[:, 0] - radii,
            centers[:, 1] - radii,
            1.0 - centers[:, 1] - radii
        ])
        
        # Pairwise non-overlap constraints: ||c_i - c_j||^2 - (r_i + r_j)^2 >= 0
        # Vectorized computation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        p_cons = dist_sq[idx_i, idx_j] - r_sum[idx_i, idx_j]**2
        
        return np.concatenate([b_cons, p_cons])

    # Variable bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # --- Initialization ---
    configs = []
    
    # 1. Hexagonal lattice pattern
    hex_pts = []
    s = 0.25
    y = s
    row = 0
    while y <= 1.0:
        x = s + (row % 2) * (s / 2)
        while x <= 1.0:
            hex_pts.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    configs.append(np.array(hex_pts[:26]))
    
    # 2. Uniform grid pattern with extra circle
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    grid_pts = np.array([(x, y) for y in grid_y for x in grid_x])
    grid_pts = np.vstack([grid_pts, [[0.5, 0.5]]])
    configs.append(grid_pts)
    
    # 3-10. Randomized perturbations of base configs
    np.random.seed(42)
    for i in range(8):
        base = configs[i % 2]
        pert = base + np.random.uniform(-0.08, 0.08, base.shape)
        configs.append(np.clip(pert, 0.05, 0.95))
        
    # --- Optimization ---
    for init_c in configs:
        r0 = np.full(n, 0.09)
        x0 = np.concatenate([init_c.flatten(), r0])
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12}
            )
            
            # Accept if finite and constraints are satisfied (allowing tiny numerical slack)
            if np.isfinite(res.fun) and np.all(constraint_func(res.x) >= -1e-5):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt = res.x[2*n:3*n]
                current_sum = np.sum(r_opt)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Fallback if optimization yields no valid result
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(n, 0.08)
        best_sum = 26 * 0.08
        
    # --- Post-processing: Ensure strict validity ---
    # Calculate the maximum safe scaling factor for radii
    scale = 1.0
    
    # Check boundary limits
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-9:
            continue
        scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
        
    # Check pairwise distance limits
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            r_sum = best_radii[i] + best_radii[j]
            if r_sum < 1e-9:
                continue
            scale = min(scale, d / r_sum)
            
    # Apply scale with a tiny safety margin for numerical precision
    best_radii *= scale * 0.99995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
