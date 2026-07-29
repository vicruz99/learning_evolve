# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000022 (state c64c9b23) state=c5da3b7c sum of radii=2.608832 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_flat, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_flat[2*n:])

def compute_constraints(vars_flat, n):
    """Compute all inequality constraints: boundary and non-overlap."""
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    c5 = dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]
    
    return np.concatenate([c1, c2, c3, c4, c5])

def run_packing():
    n = 26
    best_sum = -np.inf
    best_vars = None
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    # Prepare initial configurations
    inits = []
    
    # 1. Perturbed Hexagonal Lattices
    for seed in range(10):
        rng = np.random.default_rng(seed)
        centers = []
        y = 0.08 + rng.uniform(0, 0.02)
        row = 0
        while len(centers) < n:
            x = 0.08 + rng.uniform(0, 0.02)
            if row % 2 == 1:
                x += 0.09  # Shift for hex pattern
            while x < 0.92 and len(centers) < n:
                centers.append([x, y])
                x += 0.16 + rng.uniform(-0.01, 0.01)
            y += 0.13 + rng.uniform(-0.01, 0.01)
            row += 1
        centers = np.array(centers[:n])
        radii = np.full(n, 0.05)
        inits.append(np.concatenate([centers.flatten(), radii]))
        
    # 2. Grid Configuration
    grid_x = np.linspace(0.15, 0.85, 5)
    grid_y = np.linspace(0.15, 0.85, 5)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_c = np.column_stack([gx.flatten(), gy.flatten()])
    grid_c = np.vstack([grid_c, [0.5, 0.5]])  # 26th circle in center
    grid_r = np.full(n, 0.08)
    inits.append(np.concatenate([grid_c.flatten(), grid_r]))
    
    # 3. Random Configurations
    for seed in range(10):
        rng = np.random.default_rng(seed)
        rc = rng.uniform(0.1, 0.9, (n, 2))
        rr = np.full(n, 0.04)
        inits.append(np.concatenate([rc.flatten(), rr]))
        
    # Phase 1: Multi-start optimization
    for x0 in inits:
        try:
            res = minimize(compute_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=[cons], options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                c_vals = compute_constraints(res.x, n)
                if np.min(c_vals) > -1e-6:
                    best_sum = -res.fun
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Kick restarts to escape local minima
    if best_vars is not None:
        for k in range(50):
            x0 = best_vars.copy()
            # Decaying noise scale to refine gradually
            scale = 0.015 * (1.0 - k/55.0) + 0.001
            noise = np.random.normal(0, scale, x0.shape)
            x0 += noise
            # Ensure bounds are respected before optimization
            x0[:2*n] = np.clip(x0[:2*n], 0.01, 0.99)
            x0[2*n:] = np.clip(x0[2*n:], 0.001, 0.4)
            try:
                res = minimize(compute_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                               constraints=[cons], options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    c_vals = compute_constraints(res.x, n)
                    if np.min(c_vals) > -1e-6:
                        best_sum = -res.fun
                        best_vars = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization completely fails
    if best_vars is None:
        best_vars = np.zeros(3*n)
        for i in range(n):
            best_vars[3*i] = 0.5
            best_vars[3*i+1] = 0.5
            best_vars[3*i+2] = 0.01
        best_sum = 0.26

    # Extract and sanitize results
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    radii = np.maximum(radii, 0.0)
    centers = np.clip(centers, 0.0, 1.0)
    
    return centers, radii, float(best_sum)
