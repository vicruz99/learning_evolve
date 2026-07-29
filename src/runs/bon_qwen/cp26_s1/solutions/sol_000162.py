# sol_000162 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3daa574a) state=a9f76267 sum of radii=2.590404 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def compute_constraints(params, n):
    """
    Computes the constraint values for the optimization problem.
    Constraints must be >= 0 for feasibility.
    
    params: array of shape (3*n,) containing [x_0, y_0, r_0, ..., x_n, y_n, r_n]
    n: number of circles
    """
    # Reshape params
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i + 1]
        radii[i] = params[3*i + 2]
    
    # 1. Pairwise distance constraints: ||c_i - c_j|| >= r_i + r_j
    # Equivalent to: ||c_i - c_j|| - r_i - r_j >= 0
    
    # Vectorized computation of pairwise distances
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Construct matrix of (r_i + r_j)
    r_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Constraint values for all pairs i < j
    # We only need upper triangle to avoid redundancy and self-constraints
    # But for simplicity in vectorization, we can take all and mask, 
    # or just use the upper triangle indices.
    
    # Mask for upper triangle (strictly upper)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    constraint_vals_pairs = dists[mask] - r_sum_matrix[mask]
    
    # 2. Boundary constraints
    # r_i <= x_i  => x_i - r_i >= 0
    # r_i <= 1-x_i => 1 - x_i - r_i >= 0
    # r_i <= y_i  => y_i - r_i >= 0
    # r_i <= 1-y_i => 1 - y_i - r_i >= 0
    
    constraint_vals_boundary = np.concatenate([
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ])
    
    return np.concatenate([constraint_vals_pairs, constraint_vals_boundary])

def objective_function(params, n):
    """
    Objective function: Minimize negative sum of radii.
    """
    radii = np.array([params[3*i + 2] for i in range(n)])
    return -np.sum(radii)

def generate_initial_guess(n, method='grid'):
    """
    Generates an initial guess for centers and radii.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if method == 'grid':
        # Create a dense grid and pick points
        # Try to arrange in a rectangular grid closest to sqrt(N)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Grid spacing
        # Leave some margin for radii, e.g., 0.05
        # But for initialization, just fill [0, 1]
        x_coords = np.linspace(0.05, 0.95, cols)
        y_coords = np.linspace(0.05, 0.95, rows)
        
        idx = 0
        for y in y_coords:
            for x in x_coords:
                if idx < n:
                    centers[idx] = [x, y]
                    # Initial small radius
                    radii[idx] = 0.01
                    idx += 1
    else:
        # Random initialization
        centers = np.random.rand(n, 2)
        # Shift away from boundaries slightly
        centers = centers * 0.8 + 0.1
        radii[:] = 0.01

    # Flatten to params vector
    params = np.zeros(3 * n)
    for i in range(n):
        params[3*i] = centers[i, 0]
        params[3*i + 1] = centers[i, 1]
        params[3*i + 2] = radii[i]
        
    return params

def run_packing():
    n = 26
    
    # Bounds for parameters
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Constraint definition
    # We use a single NonlinearConstraint function that returns the array
    # The function returns values that must be >= 0 (lb=0)
    cons = NonlinearConstraint(
        fun=lambda p: compute_constraints(p, n),
        lb=0,
        ub=np.inf
    )
    
    best_params = None
    best_val = -np.inf
    
    # Try multiple restarts
    # 3 random, 2 grid
    methods = ['random'] * 3 + ['grid'] * 2
    
    for method in methods:
        # Use a seed for reproducibility in random attempts, or just random
        if method == 'random':
            np.random.seed(None) # Reset seed
        
        x0 = generate_initial_guess(n, method)
        
        try:
            res = minimize(
                objective_function,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=[cons],
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or -res.fun > best_val:
                # Check validity manually to be safe
                # SLSQP might return a point that satisfies constraints loosely?
                # The constraint function computes exact values.
                # Check if constraints are satisfied
                c_vals = compute_constraints(res.x, n)
                if np.min(c_vals) >= -1e-6: # Allow tiny numerical error
                    current_sum_radii = -res.fun
                    if current_sum_radii > best_val:
                        best_val = current_sum_radii
                        best_params = res.x
        except Exception as e:
            print(f"Optimization failed with {method}: {e}")
            continue

    if best_params is None:
        # Fallback to a simple valid packing if optimization fails
        # E.g. grid with small radii
        x0 = generate_initial_guess(n, 'grid')
        best_params = x0
        best_val = -objective_function(best_params, n)

    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_params[3*i]
        centers[i, 1] = best_params[3*i + 1]
        radii[i] = best_params[3*i + 2]
        
    sum_radii = np.sum(radii)
    
    # Final validation check (using the provided logic logic implicitly)
    # Just to ensure no NaNs or huge errors
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Return safe fallback
        centers = np.random.rand(n, 2) * 0.5 + 0.25
        radii = np.ones(n) * 0.01
        sum_radii = np.sum(radii)

    return centers, radii, sum_radii

# To allow the code to be run standalone for testing if needed, but the prompt asks for run_packing
# We can define the function and that's it.
