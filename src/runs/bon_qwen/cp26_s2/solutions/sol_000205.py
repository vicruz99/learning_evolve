# sol_000205 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=f74f4df1 sum of radii=2.557122 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Helper function to compute constraints
    # Returns a list of constraint values (should be >= 0)
    def get_constraints(vars):
        constraints = []
        # vars structure: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # Each circle is 3 variables
        
        centers = vars.reshape((n, 3))[:, :2] # (x, y)
        radii = vars.reshape((n, 3))[:, 2]    # r
        
        # 1. Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Which means: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            constraints.append(x - r)
            constraints.append(1.0 - x - r)
            constraints.append(y - r)
            constraints.append(1.0 - y - r)
            
            # Radius non-negativity (r >= 0)
            constraints.append(r)
            
        # 2. Pairwise non-overlap constraints
        # dist^2 >= (r1 + r2)^2  =>  dist^2 - (r1 + r2)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                constraint_val = dist_sq - r_sum**2
                constraints.append(constraint_val)
                
        return np.array(constraints)

    def objective(vars):
        # We want to maximize sum(r), so minimize -sum(r)
        radii = vars.reshape((n, 3))[:, 2]
        return -np.sum(radii)

    # Initialization
    # Place circles in a grid. 6 cols, 5 rows = 30 spots. We need 26.
    # We will skip 4 points to get 26.
    # Let's use a spacing that fits well. 
    # If we want r ~ 0.1, width ~ 1. 
    # Let's just place them somewhat evenly.
    
    initial_vars = np.zeros(3 * n)
    
    # Grid layout
    cols = 6
    rows = 5
    
    idx = 0
    skip_count = 0
    max_skip = 4 # Skip 4 points
    
    for r in range(rows):
        for c in range(cols):
            if skip_count < max_skip:
                # Skip some points, maybe corner ones or random?
                # Let's skip the last few
                skip_count += 1
                continue
            
            # Center position
            # Uniform distribution with margins
            x = (c + 0.5) / cols # Range 0.5/6 to 5.5/6 approx 0.08 to 0.91
            y = (r + 0.5) / rows # Range 0.5/5 to 4.5/5 approx 0.1 to 0.9
            
            # Initial radius
            r_init = 0.04 
            
            initial_vars[3*idx] = x
            initial_vars[3*idx+1] = y
            initial_vars[3*idx+2] = r_init
            idx += 1
            
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Define constraints object for scipy
    # Inequality constraints: c(x) >= 0
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    # Optimization
    # Method SLSQP
    result = minimize(
        objective, 
        initial_vars, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )
    
    if not result.success:
        # If optimization failed, return the result anyway, it might be valid
        pass

    # Extract results
    final_vars = result.x
    centers = final_vars.reshape((n, 3))[:, :2]
    radii = final_vars.reshape((n, 3))[:, 2]
    
    # Clip radii to be non-negative just in case
    radii = np.maximum(radii, 0.0)
    
    # Ensure centers are within bounds based on radii
    # The optimizer should handle this, but let's clamp to be safe against numerical errors
    # Actually, clamping might break constraints. 
    # Let's trust the optimizer but maybe do a tiny correction if needed.
    # But for the purpose of the task, we return the optimized result.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To run and validate
if __name__ == "__main__":
    # We need to simulate the environment check if we were running locally,
    # but here we just provide the function.
    # However, for testing purposes in a local script:
    # centers, radii, s = run_packing()
    # print("Sum:", s)
    # import sys
    # sys.path.append('/path/to/validation') # assuming validation is available
    # from validate_packing import validate_packing
    # print(validate_packing(centers, radii))
    pass
