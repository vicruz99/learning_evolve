# sol_000374 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b75b923f) state=e9ba61e6 sum of radii=2.558460 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_guess():
    # Hexagonal packing arrangement: 5-4-5-4-5-3 rows
    rows = [5, 4, 5, 4, 5, 3]
    r_init = 0.095
    y_spacing = r_init * np.sqrt(3)
    
    centers = []
    radii = []
    y = r_init
    
    for i, count in enumerate(rows):
        x_offset = r_init if i % 2 == 1 else 0.0
        total_width = (count - 1) * 2 * r_init
        start_x = (1.0 - total_width) / 2.0 + x_offset
        
        for _ in range(count):
            centers.append([start_x, y])
            radii.append(r_init)
            start_x += 2 * r_init
        y += y_spacing
        
    return np.array(centers), np.array(radii)

def obj_and_cons(z):
    n = 26
    centers = z[:n*2].reshape(n, 2)
    radii = z[n*2:]
    
    # Objective: minimize negative sum of radii
    f = -np.sum(radii)
    
    # Constraints collection
    cons_val = []
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    cons_val.append(centers[:, 0] - radii)
    cons_val.append(1.0 - centers[:, 0] - radii)
    cons_val.append(centers[:, 1] - radii)
    cons_val.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    triu_idx = np.triu_indices(n, k=1)
    cons_val.append(dists[triu_idx] - r_sum[triu_idx])
    
    return f, np.concatenate(cons_val)

def objective(z):
    return obj_and_cons(z)[0]

def constraint_fun(z):
    return obj_and_cons(z)[1]

def run_packing():
    # Generate initial configuration
    centers_init, radii_init = get_initial_guess()
    z0 = np.concatenate([centers_init.ravel(), radii_init])
    
    # Define bounds for variables: x,y in [0,1], r in [0.01, 0.2]
    bounds = [(0.0, 1.0) for _ in range(52)] + [(0.01, 0.2) for _ in range(26)]
    
    # Constraint dictionary for SLSQP
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    # Run optimization
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
    
    # Extract results
    z_opt = res.x
    centers = z_opt[:52].reshape(26, 2)
    radii = z_opt[52:]
    
    # Ensure strict validity (handle potential tiny numerical violations)
    min_dist_to_boundary = np.min([
        np.min(centers[:, 0] - radii),
        np.min(1.0 - centers[:, 0] - radii),
        np.min(centers[:, 1] - radii),
        np.min(1.0 - centers[:, 1] - radii)
    ])
    
    triu_idx = np.triu_indices(26, k=1)
    pairwise_slack = (np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))[triu_idx] - 
                      (radii[:, np.newaxis] + radii[np.newaxis, :])[triu_idx])
    min_dist_pairwise = np.min(pairwise_slack)
    
    min_slack = min(min_dist_to_boundary, min_dist_pairwise)
    
    # Only adjust if numerically violated beyond tolerance
    if min_slack < -1e-9:
        radii *= (1.0 + min_slack / np.max(radii))
        
    sum_r = float(np.sum(radii))
    return centers, radii, sum_r
