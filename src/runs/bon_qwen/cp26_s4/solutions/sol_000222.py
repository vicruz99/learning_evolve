# sol_000222 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 76d635d8) state=4c1c1798 sum of radii=2.618342 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

# Constants
N_CIRCLES = 26
BOUND = 1.0
TOLERANCE = 1e-12

def objective_function(params):
    """
    Objective to minimize: negative sum of radii.
    params is a flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    # Extract radii (indices 2, 5, 8, ...)
    radii = params[2::3]
    return -np.sum(radii)

def boundary_constraints(params):
    """
    Constraints for boundaries:
    r <= x <= 1-r  =>  x - r >= 0  and  x + r <= 1
    r <= y <= 1-r  =>  y - r >= 0  and  y + r <= 1
    
    Returns array of constraint values that must be >= 0.
    """
    constraints = []
    for i in range(N_CIRCLES):
        idx = i * 3
        x = params[idx]
        y = params[idx + 1]
        r = params[idx + 2]
        
        # x - r >= 0
        constraints.append(x - r)
        # 1 - (x + r) >= 0 => x + r <= 1
        constraints.append(1.0 - x - r)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - (y + r) >= 0 => y + r <= 1
        constraints.append(1.0 - y - r)
        
        # r >= 0
        constraints.append(r)
        
    return np.array(constraints)

def overlap_constraints(params):
    """
    Constraints for non-overlap:
    dist(i, j) >= r_i + r_j
    dist^2 >= (r_i + r_j)^2
    
    Returns array of constraint values that must be >= 0.
    """
    constraints = []
    for i in range(N_CIRCLES):
        idx_i = i * 3
        xi, yi, ri = params[idx_i], params[idx_i + 1], params[idx_i + 2]
        
        for j in range(i + 1, N_CIRCLES):
            idx_j = j * 3
            xj, yj, rj = params[idx_j], params[idx_j + 1], params[idx_j + 2]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            rad_sum_sq = (ri + rj)**2
            
            # We want dist_sq - rad_sum_sq >= 0
            constraints.append(dist_sq - rad_sum_sq)
            
    return np.array(constraints)

def run_packing():
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N_CIRCLES
    
    # Define constraints
    cons = []
    
    # Boundary constraints
    cons.append({
        'type': 'ineq',
        'fun': boundary_constraints
    })
    
    # Overlap constraints
    cons.append({
        'type': 'ineq',
        'fun': overlap_constraints
    })
    
    best_sum_radii = -np.inf
    best_params = None
    
    # Run multiple restarts to avoid local minima
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for _ in range(n_restarts):
        # Initialize parameters
        # Start with a grid-like initialization to help convergence, but perturbed
        params = np.zeros(3 * N_CIRCLES)
        
        # Place circles in a rough grid first
        # 5x5 grid is 25 circles, we have 26. 
        # Let's try a hexagonal-ish initialization or just random
        # Random initialization with small radii is safer for solver start
        
        # Random centers
        params[0::3] = rng.uniform(0.1, 0.9, N_CIRCLES) # x
        params[1::3] = rng.uniform(0.1, 0.9, N_CIRCLES) # y
        params[2::3] = rng.uniform(0.01, 0.05, N_CIRCLES) # r (small initial radii)
        
        # Optimize
        try:
            res = opt.minimize(
                objective_function,
                params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or (res.fun > best_sum_radii):
                # Check validity manually to be sure (numerical noise)
                current_params = res.x
                current_radii = current_params[2::3]
                current_sum = np.sum(current_radii)
                
                # Basic validation check inside loop to filter bad solutions
                valid = True
                centers = np.column_stack((current_params[0::3], current_params[1::3]))
                
                # Check bounds
                if np.any(current_radii < -TOLERANCE):
                    valid = False
                
                if np.any(centers[:, 0] - current_radii < -TOLERANCE) or \
                   np.any(centers[:, 0] + current_radii > 1 + TOLERANCE) or \
                   np.any(centers[:, 1] - current_radii < -TOLERANCE) or \
                   np.any(centers[:, 1] + current_radii > 1 + TOLERANCE):
                    valid = False
                
                # Check overlaps
                if valid:
                    for i in range(N_CIRCLES):
                        for j in range(i + 1, N_CIRCLES):
                            dist = np.linalg.norm(centers[i] - centers[j])
                            if dist < current_radii[i] + current_radii[j] - TOLERANCE:
                                valid = False
                                break
                        if not valid: break
                
                if valid and current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_params = current_params.copy()
                    
        except Exception as e:
            continue
            
    if best_params is None:
        # Fallback to a simple grid if optimization fails completely
        best_params = np.zeros(3 * N_CIRCLES)
        # 5x5 grid for 25, plus 1 small one?
        # This is just a placeholder
        idx = 0
        for r in range(5):
            for c in range(5):
                best_params[idx] = 0.1 + c * 0.2
                best_params[idx+1] = 0.1 + r * 0.2
                best_params[idx+2] = 0.1
                idx += 3
        # 26th circle
        best_params[idx] = 0.2
        best_params[idx+1] = 0.2
        best_params[idx+2] = 0.04
        best_sum_radii = 25 * 0.1 + 0.04

    centers = np.column_stack((best_params[0::3], best_params[1::3]))
    radii = best_params[2::3]
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 0)
    
    return centers, radii, float(best_sum_radii)
