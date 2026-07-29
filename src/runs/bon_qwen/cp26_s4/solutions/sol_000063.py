# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bb78642d) state=c0c77d25 sum of radii=2.549656 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N_CIRCLES = 26

def compute_constraints(v):
    """
    Computes all inequality constraints for the packing problem.
    v: 1D array of size N_CIRCLES * 3, containing [x1, y1, r1, x2, y2, r2, ...]
    Returns: 1D array of constraint values (must be >= 0)
    """
    # Reshape to (N, 3)
    pts = v.reshape(N_CIRCLES, 3)
    
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    constraints = []
    
    # 1. Boundary constraints: x - r >= 0
    constraints.append(x - r)
    
    # 2. Boundary constraints: 1 - (x + r) >= 0
    constraints.append(1.0 - (x + r))
    
    # 3. Boundary constraints: y - r >= 0
    constraints.append(y - r)
    
    # 4. Boundary constraints: 1 - (y + r) >= 0
    constraints.append(1.0 - (y + r))
    
    # 5. Non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    # Vectorized computation for all pairs
    # diff_x[i, j] = x[i] - x[j]
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2
    
    # sum_r[i, j] = r[i] + r[j]
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    sum_r_sq = sum_r**2
    
    # Constraint matrix
    cons_matrix = dist_sq - sum_r_sq
    
    # We only need upper triangular part (i < j) to avoid duplicates and self-interaction
    # Using boolean mask for upper triangle excluding diagonal
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    constraints.append(cons_matrix[mask])
    
    # Concatenate all constraints
    return np.concatenate(constraints)

def objective(v):
    """
    Objective function to minimize: -sum(radii)
    """
    pts = v.reshape(N_CIRCLES, 3)
    return -np.sum(pts[:, 2])

def run_packing():
    # Initial guess: A valid grid packing with radius 0.07
    # This provides a feasible starting point for the optimizer
    x0 = np.zeros(N_CIRCLES * 3)
    count = 0
    
    # Grid parameters
    start_pos = 0.15
    spacing = 0.15
    
    # Fill a 6x5 grid (30 spots) with 26 circles
    for r_idx in range(5):
        for c_idx in range(6):
            if count >= N_CIRCLES:
                break
            
            idx = count * 3
            x = start_pos + c_idx * spacing
            y = start_pos + r_idx * spacing
            
            # Ensure within bounds [0, 1]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            x0[idx] = x
            x0[idx+1] = y
            x0[idx+2] = 0.07 # Initial radius
            
            count += 1
        if count >= N_CIRCLES:
            break
            
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Constraints
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Run optimization
    best_x = x0
    try:
        res = opt.minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 3000, 'ftol': 1e-10}
        )
        
        if res.success:
            best_x = res.x
        else:
            # If optimization fails to converge, check if the result is feasible
            # A small tolerance allows for numerical errors
            if compute_constraints(res.x).min() >= -1e-5:
                best_x = res.x
    except Exception:
        # In case of any error, fallback to initial guess
        pass 

    # Extract results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    
    for i in range(N_CIRCLES):
        idx = i * 3
        centers[i, 0] = best_x[idx]
        centers[i, 1] = best_x[idx+1]
        radii[i] = best_x[idx+2]
        
    # Ensure radii are non-negative (safety check)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
