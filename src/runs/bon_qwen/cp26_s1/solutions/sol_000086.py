# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e9cb3956) state=b9b2b358 sum of radii=2.574641 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization
    # Create a dense grid of potential centers and select the first 26
    x_grid = np.linspace(0.05, 0.95, 6)
    y_grid = np.linspace(0.05, 0.95, 5)
    
    centers_init = []
    for y in y_grid:
        for x in x_grid:
            centers_init.append([x, y])
            if len(centers_init) == n:
                break
        if len(centers_init) == n:
            break
            
    centers_init = np.array(centers_init)
    radii_init = np.full(n, 0.05)
    
    # Flatten initial state for the optimizer: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3 * i] = centers_init[i, 0]
        x0[3 * i + 1] = centers_init[i, 1]
        x0[3 * i + 2] = radii_init[i]
        
    # 2. Objective Function
    # Minimize -sum(radii) to maximize sum(radii)
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)
        
    # 3. Constraints
    # Vector function that returns all constraint violations
    # We use squared distances: ||c_i - c_j||^2 - (r_i + r_j)^2 >= 0
    def constraints_func(x):
        centers = x.reshape(n, 3)[:, :2]
        radii = x[2::3]
        
        # Collect constraints into a single vector
        # Format: [boundary constraints..., overlap constraints...]
        c_vec = []
        
        # Boundary constraints (x - r >= 0, 1 - x - r >= 0, etc.)
        # Flattened to a 1D array
        c_vec.append(centers[:, 0] - radii)             # x >= r
        c_vec.append(1 - centers[:, 0] - radii)          # 1 - x >= r
        c_vec.append(centers[:, 1] - radii)              # y >= r
        c_vec.append(1 - centers[:, 1] - radii)          # 1 - y >= r
        
        # Overlap constraints
        # Using squared distance for smoothness: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # Iterate pairs to build vector
        overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                rad_sum_sq = (radii[i] + radii[j])**2
                overlap_constraints.append(dist_sq - rad_sum_sq)
        
        c_vec.append(np.array(overlap_constraints))
        
        return np.concatenate(c_vec)

    # Constraint bounds: All constraints must be >= 0
    # Count of constraints: 4*n boundary + n*(n-1)/2 overlap
    n_cons = 4 * n + (n * (n - 1)) // 2
    cons = NonlinearConstraint(constraints_func, lb=np.zeros(n_cons), ub=np.inf)

    # 4. Variable Bounds
    # x, y in [0, 1], r in [0, 0.5] (theoretical max radius in unit square)
    bounds = [(0, 1)] * (n * 3)
    for i in range(n):
        bounds[3 * i + 2] = (0, 0.5) # Limit radii
        
    # 5. Optimization
    # trust-constr is suitable for nonlinear constrained problems
    res = minimize(
        objective, 
        x0, 
        method='trust-constr', 
        bounds=bounds, 
        constraints=cons,
        options={'maxiter': 1000, 'verbose': 0}
    )
    
    # 6. Extract Results
    final_x = res.x.reshape(n, 3)
    centers = final_x[:, :2]
    radii = final_x[:, 2]
    
    # Ensure non-negative radii (handle numerical noise)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
