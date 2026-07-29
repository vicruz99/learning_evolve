# sol_000340 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=b729476e sum of radii=2.467727 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Objective: maximize r (minimize -r)
    def objective(vars):
        # vars structure: [x0, y0, x1, y1, ..., x25, y25, r]
        return -vars[-1]

    # Constraints function
    # Returns a vector of inequalities >= 0
    def constraint_func(vars):
        centers = vars[:2*n].reshape((n, 2))
        r = vars[-1]
        
        # Boundary constraints:
        # x >= r  => x - r >= 0
        # x <= 1-r => 1 - r - x >= 0
        # Same for y
        c_boundary = np.concatenate([
            centers[:, 0] - r,
            1 - r - centers[:, 0],
            centers[:, 1] - r,
            1 - r - centers[:, 1]
        ])
        
        # Distance constraints: ||ci - cj||^2 >= (2r)^2
        # Vectorized computation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Extract upper triangle (excluding diagonal)
        indices = np.triu_indices(n, k=1)
        dist_sq_vals = dist_sq[indices]
        
        # Constraint: dist_sq - 4r^2 >= 0
        c_dist = dist_sq_vals - 4 * r**2
        
        return np.concatenate([c_boundary, c_dist])

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n)
    bounds.append((0.0, 0.5)) 

    # Initialization with Hexagonal Packing
    # Start with a radius small enough to fit 26 circles easily
    r_init = 0.09 
    centers_init = []
    y = r_init
    row_idx = 0
    count = 0
    
    # Generate hexagonal grid points
    while count < n:
        if y > 1 - r_init + 1e-9:
            break 
            
        x = r_init
        if row_idx % 2 == 1:
            x = 2 * r_init
        
        while count < n:
            if x > 1 - r_init + 1e-9:
                break
            centers_init.append([x, y])
            count += 1
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
    
    # Pad if necessary (should not happen with r=0.09)
    while len(centers_init) < n:
        centers_init.append([0.5, 0.5])
        
    centers_init = np.array(centers_init[:n])
    vars0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    best_r = 0
    best_centers = None
    
    # Run optimization with a few restarts/perturbations
    for trial in range(3):
        if trial > 0:
            # Add small random noise to escape local minima
            np.random.seed(trial + 42)
            noise = np.random.normal(0, 0.005, size=(n, 2))
            c_noise = centers_init + noise
            # Clip to stay within reasonable bounds relative to r_init
            c_noise = np.clip(c_noise, r_init, 1 - r_init)
            current_vars = np.concatenate([c_noise.flatten(), [r_init]])
        else:
            current_vars = vars0
            
        try:
            res = minimize(objective, current_vars, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
            
            r_cand = res.x[-1]
            c_cand = res.x[:2*n].reshape((n, 2))
            
            # Strict validation
            valid = True
            
            # Check boundaries
            if np.any(c_cand - r_cand < -1e-9) or np.any(c_cand + r_cand > 1 + 1e-9):
                valid = False
            
            # Check overlaps
            if valid:
                diff = c_cand[:, np.newaxis, :] - c_cand[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                indices = np.triu_indices(n, k=1)
                min_d2 = np.min(dist_sq[indices])
                # Allow tiny numerical error
                if min_d2 < (2 * r_cand)**2 - 1e-9:
                    valid = False
            
            if valid:
                if r_cand > best_r:
                    best_r = r_cand
                    best_centers = c_cand
        except Exception as e:
            pass

    # Fallback if optimization failed
    if best_centers is None or best_r < 0.01:
        best_r = 0.05
        best_centers = np.zeros((n, 2))
        idx = 0
        # Generate a valid grid packing
        for r in range(10):
            for c in range(10):
                if idx >= n: break
                best_centers[idx] = [0.05 + c*0.1, 0.05 + r*0.1]
                idx += 1
            if idx >= n: break

    radii = np.full(26, best_r)
    return best_centers, radii, 26 * best_r
