# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5dc93b19) state=08ffc5b0 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal packing pattern
    # We generate enough centers and select the first 26.
    # Radius estimate for initialization: 0.09 (small enough to fit easily)
    r_init = 0.09
    centers = []
    
    # Hexagonal lattice generation
    # Row spacing: sqrt(3) * r
    # Col spacing: 2 * r
    # Shift odd rows by r
    row_idx = 0
    while len(centers) < n:
        y = r_init + row_idx * np.sqrt(3) * r_init
        
        # Shift x for odd rows
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init
        
        col_idx = 0
        while True:
            x = x_start + col_idx * 2 * r_init
            if x > 1 - r_init:
                break
            centers.append([x, y])
            col_idx += 1
        row_idx += 1
        
    # Trim to exactly n circles
    centers = np.array(centers[:n])
    
    # 2. Optimization: Maximize radius r (assuming equal radii)
    # Variables: x1, y1, x2, y2, ..., x26, y26, r
    # Total variables: 52 + 1 = 53
    # However, scipy optimize works with 1D arrays.
    # Let's map indices: 0..51 for coords, 52 for r.
    
    def objective(vars):
        # We want to maximize sum of radii. Since r_i = r for all i,
        # maximizing r is equivalent to maximizing sum.
        # Objective for minimizer is negative of sum.
        return -vars[52] # Maximize r

    def constraint_boundary(vars):
        # r <= x_i <= 1-r  => x_i - r >= 0  and  1 - r - x_i >= 0
        # r <= y_i <= 1-r  => y_i - r >= 0  and  1 - r - y_i >= 0
        r = vars[52]
        coords = vars[:52].reshape(n, 2)
        # Flatten constraints
        c1 = coords[:, 0] - r
        c2 = (1 - r) - coords[:, 0]
        c3 = coords[:, 1] - r
        c4 = (1 - r) - coords[:, 1]
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(vars):
        # dist(i, j) >= 2r
        # dist^2 >= 4r^2
        r = vars[52]
        coords = vars[:52].reshape(n, 2)
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((coords[i] - coords[j])**2)
                # constraint: dist_sq - 4r^2 >= 0
                constraints.append(dist_sq - 4 * r**2)
        return np.array(constraints)

    # Initial guess
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (upper bound 0.5 is safe)
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)]
    
    # Constraints
    cons = (
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    )
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            final_vars = res.x
            final_centers = final_vars[:52].reshape(n, 2)
            final_r = final_vars[52]
            final_radii = np.full(n, final_r)
            
            # Calculate sum of radii
            sum_radii = np.sum(final_radii)
            
            # Return
            return final_centers, final_radii, sum_radii
        else:
            # If optimization fails, return initial valid packing (scaled down if needed to be valid)
            # Fallback: use the initial configuration but ensure it's valid.
            # The initial config was generated with r_init=0.09 which should be valid.
            # However, coordinates might have drifted or been cut.
            # Let's just return the initial valid one with r_init.
            # Re-verify initial centers
            valid_centers = centers
            valid_radii = np.full(n, r_init)
            
            # Check validity of initial
            # (Simple check, assuming construction was correct)
            return valid_centers, valid_radii, np.sum(valid_radii)
            
    except Exception:
        # Fallback in case of errors
        return centers, np.full(n, r_init), 26 * r_init

# Helper to ensure top-level functions and no closures
# The function run_packing is self-contained.
