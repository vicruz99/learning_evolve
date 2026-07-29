# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8a979775) state=949a6626 sum of radii=2.499235 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_initial_guess(N):
    """
    Generates an initial configuration of N circles using a hexagonal packing pattern.
    Returns centers (N, 2) and radii (N,).
    """
    # Initial estimate for radius. 
    # 26 circles in 1x1 square. 
    # Hexagonal packing density allows slightly larger r than square grid.
    # r=0.09 fits well.
    r = 0.09 
    
    centers = []
    row_h = r * np.sqrt(3)
    col_w = 2 * r
    
    y = r
    row_idx = 0
    
    while len(centers) < N:
        x = r
        # Shift odd rows by r (half of 2r spacing) to create hex pattern
        if row_idx % 2 == 1:
            x = r + r 
        
        # Calculate how many circles fit in this row
        # Center positions: x, x + col_w, x + 2*col_w, ...
        # Last center must be <= 1 - r
        # x + k * col_w <= 1 - r
        # k * col_w <= 1 - 2r
        max_k = int((1 - 2*r) / col_w) + 1
        
        for k in range(max_k):
            if len(centers) >= N:
                break
            cx = x + k * col_w
            cy = y
            centers.append([cx, cy])
        
        y += row_h
        row_idx += 1
        
    return np.array(centers), np.full(N, r)

def define_constraints_and_objective(N):
    """
    Defines the objective function and constraint function for the optimizer.
    """
    def objective(v):
        # Maximize sum of radii => Minimize negative sum
        # Radii are stored in the last N elements of v
        return -np.sum(v[2*N:])

    def constr_func(v):
        # Extract centers and radii
        # v[0:2N] are centers (x1, y1, x2, y2, ...)
        # v[2N:] are radii
        centers = v[:2*N].reshape(N, 2)
        radii = v[2*N:]
        
        # 1. Boundary Constraints
        # Circle i inside [0,1]^2 means:
        # x_i >= r_i  => x_i - r_i >= 0
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        # y_i >= r_i  => y_i - r_i >= 0
        # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
        
        c_bound_x_min = centers[:, 0] - radii
        c_bound_x_max = 1.0 - centers[:, 0] - radii
        c_bound_y_min = centers[:, 1] - radii
        c_bound_y_max = 1.0 - centers[:, 1] - radii
        
        # 2. Non-overlap Constraints
        # Distance between center i and j must be >= r_i + r_j
        # Squared: ||c_i - c_j||^2 >= (r_i + r_j)^2
        # ||c_i - c_j||^2 - (r_i + r_j)^2 >= 0
        
        # Vectorized distance calculation
        # diff shape: (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Radii sum squared
        radii_sum_sq = (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2
        
        # Constraint values: dist_sq - radii_sum_sq
        # We only need upper triangle (i < j) to avoid duplicates and self-comparison
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = dists_sq[mask] - radii_sum_sq[mask]
        
        # Concatenate all constraints
        # Each must be >= 0
        return np.concatenate([c_bound_x_min, c_bound_x_max, c_bound_y_min, c_bound_y_max, c_overlap])

    cons = {'type': 'ineq', 'fun': constr_func}
    return objective, cons

def run_packing():
    N = 26
    
    # 1. Initialization
    centers_init, radii_init = generate_initial_guess(N)
    
    # Add small random noise to avoid symmetric local minima
    np.random.seed(123)
    noise = np.random.randn(N, 2) * 0.0005
    centers_init = centers_init + noise
    # Clip to ensure valid start (though constraints handle it, good practice)
    centers_init = np.clip(centers_init, 1e-6, 1 - 1e-6)
    
    # Flatten to 1D vector for optimizer
    # Format: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # 2. Bounds
    # x, y in [0, 1]
    # r in [0, 0.5] (cannot be larger than 0.5 in unit square)
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # 3. Define Problem
    objective, cons = define_constraints_and_objective(N)
    
    # 4. Optimize
    # SLSQP is suitable for constrained non-linear problems
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'disp': False
    }
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options=options)
        
        if res.success or (res.fun < 0 and np.all(cons['fun'](res.x) >= -1e-9)):
            # Extract results
            best_centers = res.x[:2*N].reshape(N, 2)
            best_radii = res.x[2*N:]
            sum_radii = np.sum(best_radii)
            
            # Basic sanity check before returning
            # If any radius is negative (due to numerical issues), fix it
            best_radii = np.maximum(best_radii, 0.0)
            
            return best_centers, best_radii, sum_radii
        else:
            # If optimization failed, return the initial guess (which is valid)
            # But we should ensure the initial guess is strictly valid
            # Our generation logic ensures validity for r=0.09
            return centers_init, radii_init, np.sum(radii_init)
            
    except Exception:
        # Fallback
        return centers_init, radii_init, np.sum(radii_init)
