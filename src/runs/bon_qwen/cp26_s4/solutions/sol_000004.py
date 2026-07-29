# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d058a929) state=e2b7680f sum of radii=2.500702 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses numerical optimization to find the maximum equal radius packing.
    """
    n = 26
    
    # 1. Initial Guess: Hexagonal Lattice
    # We place circles in a staggered grid to mimic hexagonal packing.
    # We estimate a radius slightly less than optimal to ensure initial validity.
    # For 26 circles, a rough guess for r is 0.09.
    # Spacing approx 2r.
    
    centers = []
    y_step = 0.17  # Vertical spacing (approx sqrt(3)/2 * diameter)
    y_curr = 0.12  # Start slightly away from edge
    
    row = 0
    while len(centers) < n:
        # Horizontal spacing approx diameter. Stagger odd rows.
        x_step = 0.18
        offset = x_step / 2 if row % 2 == 1 else 0
        
        # Start x position, ensuring we stay within bounds roughly
        x_curr = 0.12 + offset
        
        while x_curr <= 0.90 and len(centers) < n:
            centers.append([x_curr, y_curr])
            x_curr += x_step
            
        y_curr += y_step
        row += 1
        
    centers = np.array(centers[:n])
    
    # 2. Define Objective and Constraints for scipy.optimize
    # We optimize variables: [x1, y1, ..., x26, y26, r]
    # Total variables: 2*n + 1
    
    def objective(vars):
        # We want to maximize r, so minimize -r
        return -vars[-1]

    def get_centers_and_r(vars):
        c = vars[:-1].reshape(n, 2)
        r = vars[-1]
        return c, r

    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # This can be written as:
    # x_i - r >= 0
    # 1 - x_i - r >= 0
    # y_i - r >= 0
    # 1 - y_i - r >= 0
    # r >= 0.001 (small positive radius)
    
    constraints = []
    
    # Add boundary constraints
    for i in range(n):
        # x_i - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[2*idx] - v[-1]
        })
        # 1 - x_i - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[2*idx] - v[-1]
        })
        # y_i - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[2*idx+1] - v[-1]
        })
        # 1 - y_i - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[2*idx+1] - v[-1]
        })

    # Non-overlap constraints: ||c_i - c_j|| >= 2r
    # Squared: (x_i-x_j)^2 + (y_i-y_j)^2 >= 4r^2
    # => (x_i-x_j)^2 + (y_i-y_j)^2 - 4r^2 >= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - 4*(v[-1])**2
            })

    # 3. Setup Initial Variables
    # Flatten centers and append initial radius guess
    # Initial radius guess based on the grid spacing used
    initial_r = 0.09 
    x0 = np.concatenate([centers.flatten(), [initial_r]])
    
    # Bounds for variables
    # x, y in [0, 1] (actually tighter, but [0,1] is safe)
    # r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)]
    
    # 4. Optimization
    # SLSQP handles inequality constraints well.
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )
    
    # 5. Extract Solution
    if result.success or result.fun > -0.5: # r < 0.5
        final_vars = result.x
        centers_opt = final_vars[:-1].reshape(n, 2)
        radius_opt = final_vars[-1]
    else:
        # Fallback if optimization fails (unlikely with good start)
        # Return the initial configuration with a safe radius
        centers_opt = centers
        # Calculate max feasible radius for initial config
        min_dist = 1.0
        for i in range(n):
            # Boundary dist
            d_bound = min(centers_opt[i,0], 1-centers_opt[i,0], 
                          centers_opt[i,1], 1-centers_opt[i,1])
            if d_bound < min_dist:
                min_dist = d_bound
            
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                if dist < min_dist:
                    min_dist = dist
        
        radius_opt = min_dist / 2.0

    radii_opt = np.full(n, radius_opt)
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii

# Ensure function is callable
if __name__ == "__main__":
    # Just to verify structure if run directly
    pass
