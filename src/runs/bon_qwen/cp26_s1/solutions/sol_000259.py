# sol_000259 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3be09fa9) state=aea04da6 sum of radii=2.540000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Assuming validate_packing is defined in the environment as per instructions.
# If running this code block standalone, you would need the validate_packing function defined.

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    def get_constraint_slacks(x):
        """
        Computes the slack for all constraints.
        Returns an array where all values should be >= 0 for a valid packing.
        Structure:
        - [0:n]: x_i - r_i >= 0
        - [n:2n]: 1 - x_i - r_i >= 0
        - [2n:3n]: y_i - r_i >= 0
        - [3n:4n]: 1 - y_i - r_i >= 0
        - [4n: ...]: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        """
        # Number of distance constraints: n*(n-1)/2
        num_dist = n * (n - 1) // 2
        slacks = np.zeros(4 * n + num_dist)
        
        # Boundary constraints
        for i in range(n):
            idx = 3 * i
            xi, yi, ri = x[idx], x[idx+1], x[idx+2]
            
            slacks[i] = xi - ri
            slacks[n + i] = 1.0 - xi - ri
            slacks[2 * n + i] = yi - ri
            slacks[3 * n + i] = 1.0 - yi - ri
            
        # Distance constraints
        dist_idx = 4 * n
        for i in range(n):
            idx_i = 3 * i
            for j in range(i + 1, n):
                idx_j = 3 * j
                
                dx = x[idx_i] - x[idx_j]
                dy = x[idx_i + 1] - x[idx_j + 1]
                dr = x[idx_i + 2] + x[idx_j + 2]
                
                dist_sq = dx**2 + dy**2
                slacks[dist_idx] = dist_sq - dr**2
                dist_idx += 1
                
        return slacks

    def objective(x):
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(x[2::3])

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    # We use loose bounds here, letting the constraints enforce the geometry
    bounds = [(0.0, 1.0)] * (3 * n)

    constraints = ({
        'type': 'ineq',
        'fun': get_constraint_slacks
    })

    def generate_initial_guess(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Start with a dense 5x5 grid configuration
        grid_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [grid_coords[i], grid_coords[j]]
                radii[idx] = 0.1
                idx += 1
        
        # Place the 26th circle in a gap
        centers[25] = [0.2, 0.2]
        radii[25] = 0.04
        
        # Add jitter to break symmetry
        jitter = np.random.normal(0, 0.01, size=(n, 2))
        centers = centers + jitter
        # Clip to stay away from immediate boundary issues
        centers = np.clip(centers, 0.05, 0.95)
        
        # Flatten to optimization vector [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3 * i] = centers[i, 0]
            x0[3 * i + 1] = centers[i, 1]
            x0[3 * i + 2] = radii[i]
            
        return x0

    best_x = None
    best_sum_radii = -1.0
    
    # Run multiple restarts with different seeds
    for seed in [42, 123, 456, 789, 101]:
        x0 = generate_initial_guess(seed)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                # Extract solution
                centers_sol = np.zeros((n, 2))
                radii_sol = np.zeros(n)
                for i in range(n):
                    centers_sol[i] = [res.x[3 * i], res.x[3 * i + 1]]
                    radii_sol[i] = res.x[3 * i + 2]
                
                # Validate using the provided function
                # Note: validate_packing is assumed to be available in the global scope
                # If running this code in a fresh environment, you'd need to define it.
                # Here we assume the environment has it.
                # To be safe, we can try to call it.
                try:
                    if validate_packing(centers_sol, radii_sol):
                        current_sum = np.sum(radii_sol)
                        if current_sum > best_sum_radii:
                            best_sum_radii = current_sum
                            best_x = res.x
                except NameError:
                    # Fallback if validate_packing is not defined in this specific scope
                    # (though the prompt implies it is)
                    pass
        except Exception:
            continue

    if best_x is not None:
        centers_final = np.zeros((n, 2))
        radii_final = np.zeros(n)
        for i in range(n):
            centers_final[i] = [best_x[3 * i], best_x[3 * i + 1]]
            radii_final[i] = best_x[3 * i + 2]
        return centers_final, radii_final, np.sum(radii_final)
    
    # Fallback to a valid grid packing if optimization fails
    centers_fallback = np.zeros((n, 2))
    radii_fallback = np.zeros(n)
    grid_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    idx = 0
    for i in range(5):
        for j in range(5):
            centers_fallback[idx] = [grid_coords[i], grid_coords[j]]
            radii_fallback[idx] = 0.1
            idx += 1
    centers_fallback[25] = [0.2, 0.2]
    radii_fallback[25] = 0.04
    return centers_fallback, radii_fallback, np.sum(radii_fallback)
