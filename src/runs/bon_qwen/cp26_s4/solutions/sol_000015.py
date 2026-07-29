# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=c0cca912 sum of radii=2.610399 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars):
    """
    Computes inequality constraints for the circle packing problem.
    Returns an array of constraint values that must be >= 0.
    """
    n = 26
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Boundary constraints: x - r >= 0, 1 - r - x >= 0, y - r >= 0, 1 - r - y >= 0
    bnd = np.zeros(4*n)
    bnd[0::4] = centers[:, 0] - radii
    bnd[1::4] = 1.0 - radii - centers[:, 0]
    bnd[2::4] = centers[:, 1] - radii
    bnd[3::4] = 1.0 - radii - centers[:, 1]
    
    # Pairwise separation constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    pair_cons = dist_sq - r_sum**2
    
    # Extract only the upper triangle to avoid duplicates
    triu_idx = np.triu_indices(n, k=1)
    p = pair_cons[triu_idx]
    
    return np.concatenate([bnd, p])

def constraint_fun_26(vars):
    return compute_constraints(vars)

def obj_fun_26(vars):
    # Maximize sum of radii <=> Minimize negative sum
    return -np.sum(vars[52:])

def get_initial_configs():
    """Generates multiple diverse initial configurations for optimization."""
    n = 26
    configs = []
    
    # Config 1: 6x6 grid (first 26 points)
    x = np.linspace(0.1, 0.9, 6)
    y = np.linspace(0.1, 0.9, 6)
    pts = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    configs.append(np.concatenate([pts[:n].flatten(), np.full(n, 0.03)]))
    
    # Config 2: 5x5 grid + 1 in center
    x5 = np.linspace(0.15, 0.85, 5)
    y5 = np.linspace(0.15, 0.85, 5)
    pts5 = np.array(np.meshgrid(x5, y5)).T.reshape(-1, 2)
    pts5_add = np.vstack([pts5, [0.5, 0.5]])
    configs.append(np.concatenate([pts5_add.flatten(), np.full(n, 0.04)]))
    
    # Config 3: Hexagonal lattice approximation
    # Rows pattern: 6, 5, 6, 5, 4 circles
    counts = [6, 5, 6, 5, 4]
    y_vals = np.linspace(0.15, 0.85, 5)
    centers = []
    for r, count in enumerate(counts):
        shift = 0.1 if r % 2 == 1 else 0.0
        # Adjust x range to fit within [0.05, 0.95] roughly
        x_start = 0.05 + shift
        x_end = 0.95 - shift
        if count > 1:
            x_vals = np.linspace(x_start, x_end, count)
        else:
            x_vals = [0.5]
        for x in x_vals:
            centers.append([x, y_vals[r]])
    centers = np.array(centers[:n])
    configs.append(np.concatenate([centers.flatten(), np.full(n, 0.03)]))
    
    # Config 4 & 5: Random feasible starts
    np.random.seed(42)
    for _ in range(2):
        c = np.random.rand(n*2) * 0.8 + 0.1
        r = np.random.rand(n) * 0.05 + 0.02
        configs.append(np.concatenate([c, r]))
        
    return configs

def run_packing():
    n = 26
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, 0.5)]*n
    cons = {'type': 'ineq', 'fun': constraint_fun_26}
    
    best_sol = None
    best_sum = -np.inf
    configs = get_initial_configs()
    
    # Optimize from each initial configuration
    for x0 in configs:
        try:
            res = minimize(obj_fun_26, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_sol = res
        except Exception:
            continue
            
    # Fallback in case all optimizations failed
    if best_sol is None:
        x0 = configs[0]
        best_sol = minimize(obj_fun_26, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
    vars_opt = best_sol.x
    centers = vars_opt[:2*n].reshape(n, 2)
    radii = vars_opt[2*n:]
    
    # Ensure radii are strictly non-negative
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
