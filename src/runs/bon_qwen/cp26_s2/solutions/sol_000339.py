# sol_000339 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=b44d5bf2 sum of radii=2.604978 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Fixed constants
N_CIRCLES = 26

def generate_initial_centers():
    """Generates a perturbed hexagonal grid of centers for 26 circles."""
    centers = []
    r_guess = 0.09
    
    # Generate rows
    for row_idx in range(10):  # More than enough rows
        y = r_guess + row_idx * r_guess * np.sqrt(3)
        if y + r_guess > 1.0:
            break
            
        # Shift x for odd rows to create hexagonal packing
        x_start = 2 * r_guess if row_idx % 2 == 1 else r_guess
        
        x = x_start
        while x + r_guess <= 1.0:
            if len(centers) < N_CIRCLES:
                centers.append([x, y])
            x += 2 * r_guess
            
    # If we generated fewer than 26 (unlikely with this density), pad with random
    while len(centers) < N_CIRCLES:
        centers.append([np.random.uniform(0, 1), np.random.uniform(0, 1)])
    
    # If we generated more, truncate
    centers = centers[:N_CIRCLES]
    
    return np.array(centers)

def objective(variables):
    """Negative sum of radii (to minimize)."""
    r = variables[2 * N_CIRCLES : 3 * N_CIRCLES]
    return -np.sum(r)

def constraints(variables):
    """Computes vector of constraint violations."""
    x = variables[:N_CIRCLES]
    y = variables[N_CIRCLES : 2 * N_CIRCLES]
    r = variables[2 * N_CIRCLES : 3 * N_CIRCLES]
    
    constraints_list = []
    
    # Pairwise non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    # Vectorized calculation
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist_sq = x_diff**2 + y_diff**2
    
    r_sum = r[:, None] + r[None, :]
    overlap_term = dist_sq - r_sum**2
    
    # Extract upper triangle (unique pairs)
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    pairwise_constraints = overlap_term[mask]
    constraints_list.append(pairwise_constraints)
    
    # Boundary constraints
    constraints_list.append(x - r)          # x - r >= 0
    constraints_list.append(1 - x - r)      # (1 - x) - r >= 0
    constraints_list.append(y - r)          # y - r >= 0
    constraints_list.append(1 - y - r)      # (1 - y) - r >= 0
    
    # Combine into a single array
    return np.concatenate(constraints_list)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initialize variables: [x1..x26, y1..y26, r1..r26]
    init_centers = generate_initial_centers()
    r_init = np.full(N_CIRCLES, 0.1) # Initial radius guess
    
    x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], r_init])
    
    # 2. Define bounds for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 0.5)] * N_CIRCLES
    
    # 3. Define constraints for SLSQP
    # SLSQP requires a single constraint function or dict
    cons = ({'type': 'ineq', 'fun': constraints})
    
    # 4. Run Optimization
    # Using method 'SLSQP' for handling non-linear constraints
    # tol is set to a reasonable value to prevent endless computation
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    
    # 5. Extract solution
    optimal_vars = res.x
    centers = np.column_stack((optimal_vars[:N_CIRCLES], optimal_vars[N_CIRCLES:2 * N_CIRCLES]))
    radii = optimal_vars[2 * N_CIRCLES:3 * N_CIRCLES]
    
    # 6. Safety Scaling
    # The optimizer might find a boundary solution with tiny numerical violations.
    # Scaling radii down slightly ensures strict adherence to the 1e-12 tolerance in validation.
    safety_factor = 0.995
    radii = radii * safety_factor
    
    # 7. Final validation check (internal)
    # Recalculate sum
    sum_radii = float(np.sum(radii))
    
    # Note: The validation function provided in the prompt is external.
    # We rely on the optimization and scaling to satisfy it.
    
    return centers, radii, sum_radii

# Helper functions required to be top-level as per prompt
# (They are defined above)
