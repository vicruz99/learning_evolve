import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    n = 26
    
    # --- 1. Initialization: 5x5 Grid with an extra circle ---
    # A 5x5 grid of r=0.1 fits 25 circles perfectly.
    # We add a 26th circle and start an optimization to expand them.
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i * 0.2, 0.1 + j * 0.2])
            radii.append(0.1)
    # Add 26th circle in a gap (centered at 0.2, 0.2) with small initial radius
    centers.append([0.2, 0.2])
    radii.append(0.01)

    centers = np.array(centers)
    radii = np.array(radii)

    # --- 2. Combined Optimization (SLSQP) ---
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    def to_flat(vars_tuple):
        return np.hstack([v.flatten() for v in vars_tuple])

    def from_flat(flat, n):
        vars_list = [flat[i*3:(i+1)*3] for i in range(n)]
        return vars_list

    def objective(flat):
        # Maximize sum of radii -> Minimize negative sum
        radii = from_flat(flat, n)
        return -sum(r[2] for r in radii)

    def dist_constraint(flat, i, j):
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        v_i = from_flat(flat, n)[i]
        v_j = from_flat(flat, n)[j]
        x_diff = v_i[0] - v_j[0]
        y_diff = v_i[1] - v_j[1]
        r_sum = v_i[2] + v_j[2]
        return (x_diff**2 + y_diff**2) - r_sum**2

    def wall_constraint(flat, i):
        v = from_flat(flat, n)[i]
        x, y, r = v
        # Return a vector of 4 constraints: x>=r, 1-x>=r, y>=r, 1-y>=r
        return np.array([x - r, 1 - x - r, y - r, 1 - y - r])

    constraints = []
    
    # Distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda f, i=i, j=j: dist_constraint(f, i, j)
            })
            
    # Wall constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda f, i=i: wall_constraint(f, i)
        })

    # Bounds
    # r >= 0, x, y in [0, 1]
    bounds = []
    for _ in range(n):
        bounds.extend([
            (0, 1), # x
            (0, 1), # y
            (0, 1)  # r
        ])

    initial_flat = to_flat((centers, radii))
    
    # Run optimization
    result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})

    opt_vars = from_flat(result.x, n)
    opt_centers = np.array([[v[0], v[1]] for v in opt_vars])
    opt_radii = np.array([v[2] for v in opt_vars])

    # --- 3. Position Refinement (BFGS with Penalty) ---
    # Fix radii to the optimized sum and optimize centers to minimize overlap.
    # This helps pack them tighter to potentially allow larger radii later.
    
    def penalty_function(flat_centers):
        # flat_centers is 2D array reshaped to 1D
        centers = flat_centers.reshape(-1, 2)
        total_penalty = 0.0
        
        # Circle-Circle overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                r_sum = opt_radii[i] + opt_radii[j]
                if dist < r_sum:
                    total_penalty += (r_sum - dist) ** 2
        
        # Wall overlaps
        for i in range(n):
            c = centers[i]
            r = opt_radii[i]
            dist_left = c[0] - r
            dist_right = 1 - c[0] - r
            dist_bottom = c[1] - r
            dist_top = 1 - c[1] - r
            
            for d in [dist_left, dist_right, dist_bottom, dist_top]:
                if d < 0:
                    total_penalty += d ** 2
                    
        return total_penalty

    # Try to refine centers if there are any overlaps
    current_penalty = penalty_function(opt_centers.flatten())
    if current_penalty > 1e-12:
        result_pos = minimize(penalty_function, opt_centers.flatten(), method='BFGS')
        opt_centers = result_pos.x.reshape(-1, 2)

    # Recalculate sum
    sum_radii = np.sum(opt_radii)

    return opt_centers, opt_radii, sum_radii