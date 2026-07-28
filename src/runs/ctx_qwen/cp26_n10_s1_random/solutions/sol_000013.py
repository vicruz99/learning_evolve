# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f64c520b) state=02ce6818 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for number of circles
N_CIRCLES = 26

def objective(vars):
    """
    Objective function to minimize: -sum(radii)
    vars shape: (78,) where vars[3i:3i+3] = [x_i, y_i, r_i]
    """
    r_sum = 0.0
    for i in range(N_CIRCLES):
        r_sum += vars[3 * i + 2]
    return -r_sum

def boundary_constraints(vars):
    """
    Returns array of boundary constraint values >= 0.
    Constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    """
    cons = np.zeros(4 * N_CIRCLES)
    idx = 0
    for i in range(N_CIRCLES):
        x = vars[3 * i]
        y = vars[3 * i + 1]
        r = vars[3 * i + 2]
        
        cons[idx] = x - r
        cons[idx + 1] = 1.0 - x - r
        cons[idx + 2] = y - r
        cons[idx + 3] = 1.0 - y - r
        idx += 4
    return cons

def separation_constraints(vars):
    """
    Returns array of separation constraint values >= 0.
    Constraint: dist^2 - (r_i + r_j)^2 >= 0
    """
    n = N_CIRCLES
    num_pairs = n * (n - 1) // 2
    cons = np.zeros(num_pairs)
    idx = 0
    
    # Extract centers and radii for vectorized ops if possible, 
    # but loop is clearer and fast enough for N=26
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((centers[i] - centers[j])**2)
            min_dist_sq = (radii[i] + radii[j])**2
            cons[idx] = dist_sq - min_dist_sq
            idx += 1
    return cons

def run_packing():
    # Initialize configuration: Hexagonal packing
    # Layout: 6 rows with pattern 5, 4, 5, 4, 5, 4 (Total 27, take 26)
    # This mimics a dense packing structure.
    
    r_init = 0.08
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    rows_pattern = [5, 4, 5, 4, 5, 4]
    centers_list = []
    
    y_curr = r_init
    count = 0
    for row_idx, num_circles in enumerate(rows_pattern):
        # Shift x for staggered rows
        x_start = r_init + (row_idx % 2) * r_init
        for k in range(num_circles):
            if count >= N_CIRCLES:
                break
            x = x_start + k * dx
            centers_list.append([x, y_curr])
            count += 1
        y_curr += dy
        if count >= N_CIRCLES:
            break
            
    centers = np.array(centers_list)
    radii = np.full(N_CIRCLES, r_init)
    
    # Build initial vector
    x0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]
        
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': separation_constraints}
    ]
    
    # Run optimization
    best_result = None
    best_sum = -np.inf
    
    # Try a few random perturbations to avoid local minima
    # 1. Optimized Hexagonal
    # 2. Randomized valid positions
    
    attempts = [np.copy(x0)]
    
    # Generate a few random valid attempts
    for _ in range(3):
        # Random centers, small radius
        rand_centers = np.random.rand(N_CIRCLES, 2)
        # Shift to ensure boundary clearance for small r
        rand_centers = rand_centers * 0.8 + 0.1 
        rand_radii = np.full(N_CIRCLES, 0.02)
        
        rand_vec = np.zeros(3 * N_CIRCLES)
        for i in range(N_CIRCLES):
            rand_vec[3 * i] = rand_centers[i, 0]
            rand_vec[3 * i + 1] = rand_centers[i, 1]
            rand_vec[3 * i + 2] = rand_radii[i]
        attempts.append(rand_vec)
        
    for x_init in attempts:
        try:
            res = minimize(objective, x_init, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < 0: # fun is -sum_r, so more negative is better
                current_sum = -res.fun
                # Verify feasibility with a strict check
                # SLSQP might allow slight violations due to tolerance
                # We check constraints manually
                x_final = res.x
                c_final = x_final[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_final = x_final[2*N_CIRCLES:]
                
                # Check validity manually
                is_valid = True
                # Boundary check
                for i in range(N_CIRCLES):
                    if c_final[i, 0] < r_final[i] - 1e-6 or c_final[i, 0] > 1 - r_final[i] + 1e-6:
                        is_valid = False
                        break
                    if c_final[i, 1] < r_final[i] - 1e-6 or c_final[i, 1] > 1 - r_final[i] + 1e-6:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check overlaps
                    for i in range(N_CIRCLES):
                        for j in range(i + 1, N_CIRCLES):
                            dist = np.linalg.norm(c_final[i] - c_final[j])
                            if dist < r_final[i] + r_final[j] - 1e-6:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                
                if is_valid and current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
                    
        except Exception:
            continue

    if best_result is None:
        # Fallback to initial attempt if optimization failed
        best_result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
    # Extract final result
    x_final = best_result.x
    centers = x_final[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = x_final[2 * N_CIRCLES:]
    
    # Final cleanup to ensure strict bounds (clamp slightly if needed)
    # Although constraints should handle it, numerical noise might exist.
    # However, validation function has 1e-12 tolerance.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
