# sol_000366 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c37d7b2f) state=573670dd sum of radii=2.626697 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(z):
    """
    Computes all inequality constraints for the packing problem.
    Returns an array where each element must be >= 0.
    """
    n = 26
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:]
    
    # Boundary constraints: 4 per circle
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_bound = np.zeros(4 * n)
    c_bound[:n] = x - r
    c_bound[n:2*n] = 1.0 - x - r
    c_bound[2*n:3*n] = y - r
    c_bound[3*n:4*n] = 1.0 - y - r
    
    # Overlap constraints: 1 per pair
    m = n * (n - 1) // 2
    c_ov = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            c_ov[idx] = dx * dx + dy * dy - (r[i] + r[j]) ** 2
            idx += 1
            
    return np.concatenate([c_bound, c_ov])

def objective_func(z):
    """Objective: minimize negative sum of radii."""
    return -np.sum(z[52:])

def get_initial_guess(n, seed):
    """Generates an initial guess based on a hexagonal-like grid with random perturbation."""
    rng = np.random.default_rng(seed)
    
    pts = []
    # Hexagonal arrangement: alternating row lengths 5, 6, 5, 6, 4 -> total 26
    row_counts = [5, 6, 5, 6, 4]
    row_idx = 0
    
    for i in range(5):
        count = row_counts[i]
        y_coord = 0.15 + i * 0.19
        for j in range(count):
            x_coord = 0.15 + j * 0.16
            if i % 2 == 1:
                x_coord += 0.08
            # Add small random jitter to break symmetry
            x_coord += rng.uniform(-0.02, 0.02)
            y_coord += rng.uniform(-0.02, 0.02)
            pts.append((max(0.05, min(0.95, x_coord)), max(0.05, min(0.95, y_coord))))
            
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i] = pts[i][0]
        x0[n + i] = pts[i][1]
        x0[2*n + i] = 0.04  # Initial radius small enough to satisfy constraints
        
    return x0

def run_packing():
    n = 26
    bounds = [(0.0, 1.0) for _ in range(2 * n)] + [(0.0, 1.0) for _ in range(n)]
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_res = None
    best_sum_radii = -np.inf
    
    # Run optimization from multiple starting points to escape local minima
    seeds = [0, 42, 123, 2023, 999]
    for seed in seeds:
        x0 = get_initial_guess(n, seed)
        try:
            res = minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_res = res
        except Exception:
            continue
            
    # Fallback if all failed
    if best_res is None:
        x0 = get_initial_guess(n, 0)
        best_res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
    x_opt = best_res.x
    centers = np.column_stack((x_opt[:n], x_opt[n:2*n]))
    radii = np.maximum(x_opt[2*n:], 0.0)
    
    # Final sanity clamp to ensure strict non-negativity and boundary respect numerically
    radii = np.maximum(radii, 1e-9)
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    return centers, radii, float(np.sum(radii))
