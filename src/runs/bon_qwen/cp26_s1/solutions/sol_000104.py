# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=cd7f97a3 sum of radii=2.482243 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.ndarray of shape (26, 2)
        radii: np.ndarray of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initial Guess ---
    # We start with a grid distribution to avoid bad local minima from random initialization.
    # A 6x5 grid gives 30 positions, we take 26.
    # x coordinates: 1/12, 3/12, ..., 11/12 (centers of 6 bins)
    # y coordinates: 1/10, 3/10, ..., 9/10 (centers of 5 bins)
    
    x_coords = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])
    y_coords = np.array([1/10, 3/10, 5/10, 7/10, 9/10])
    
    centers_init = []
    for y in y_coords:
        for x in x_coords:
            centers_init.append([x, y])
            if len(centers_init) >= n:
                break
        if len(centers_init) >= n:
            break
            
    centers_init = np.array(centers_init[:n])
    
    # Small initial radii to ensure valid starting configuration
    radii_init = np.full(n, 0.02)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # --- 2. Bounds ---
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((0.0, 0.5)) # r_i (cannot exceed 0.5 in unit square)
        
    # --- 3. Constraint Function ---
    # Computes margins: margin >= 0 implies valid packing.
    # Margins include:
    # 1. Distance to left boundary - r_i >= 0  => x_i - r_i
    # 2. Distance to right boundary - r_i >= 0 => 1 - x_i - r_i
    # 3. Distance to bottom boundary - r_i >= 0 => y_i - r_i
    # 4. Distance to top boundary - r_i >= 0 => 1 - y_i - r_i
    # 5. Distance between centers (i,j) - (r_i + r_j) >= 0
    
    def constraints_func(x_vec):
        # Unpack variables
        x = x_vec[0::3]
        y = x_vec[1::3]
        r = x_vec[2::3]
        
        margins = []
        
        # Boundary constraints (vectorized)
        margins.append(x - r)
        margins.append(1.0 - x - r)
        margins.append(y - r)
        margins.append(1.0 - y - r)
        
        # Overlap constraints
        # Compute pairwise distances
        # Using broadcasting to compute distance matrix
        # x_diff shape (n, n), y_diff shape (n, n)
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        
        # Distance squared to avoid sqrt for comparison? 
        # But constraint is linear in r: dist - (r_i + r_j) >= 0.
        # So we need dist.
        dists = np.sqrt(x_diff**2 + y_diff**2)
        
        # Sum of radii matrix
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Margin matrix: dists - r_sum
        overlap_margins = dists - r_sum
        
        # We only need constraints for i < j (upper triangle excluding diagonal)
        # Diagonal is 0 - 2r_i < 0, which is fine as it's not a constraint between distinct circles.
        # We extract upper triangle indices.
        idx = np.triu_indices(n, k=1)
        overlap_vals = overlap_margins[idx]
        margins.append(overlap_vals)
        
        return np.concatenate(margins)

    # Define the nonlinear constraint object
    # lower_bound=0 means all margins must be >= 0
    nl_con = NonlinearConstraint(constraints_func, lb=0.0, ub=np.inf)
    
    # --- 4. Optimization ---
    # Objective: Minimize -sum(r_i)
    def objective(x_vec):
        r = x_vec[2::3]
        return -np.sum(r)
    
    # Run optimization
    # Trust-constr is suitable for problems with bounds and nonlinear constraints
    res = minimize(
        objective, 
        x0, 
        method='trust-constr', 
        bounds=bounds, 
        constraints=nl_con,
        options={'verbose': 0, 'maxiter': 2000}
    )
    
    if not res.success:
        # If optimization failed, try to return the best found solution
        # or fall back to initial guess (though initial guess is valid)
        pass

    # Extract results
    x_opt = res.x[0::3]
    y_opt = res.x[1::3]
    r_opt = res.x[2::3]
    
    centers = np.column_stack((x_opt, y_opt))
    radii = r_opt
    
    # Final sanity check and cleaning (clip tiny negative radii if any due to numerical noise)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
