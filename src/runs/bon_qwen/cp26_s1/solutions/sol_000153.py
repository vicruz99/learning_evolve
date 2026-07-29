# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bb08abb) state=ab0fe566 sum of radii=2.361441 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Constants
N_CIRCLES = 26
PENALTY_WEIGHT = 10000.0

def objective_function(params, n, weight):
    """
    Objective function to minimize:
    - Sum of radii (to maximize sum)
    + Penalty for boundary violations
    + Penalty for overlap violations
    """
    # Extract variables
    # params layout: [x0...xn-1, y0...yn-1, r0...rn-1]
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # 1. Primary Objective: Maximize sum of radii => Minimize -sum(r)
    obj = -np.sum(r)
    
    # 2. Boundary Penalties
    # Constraint: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    dist_left = x
    dist_right = 1.0 - x
    dist_bottom = y
    dist_top = 1.0 - y
    
    dist_to_boundary = np.minimum(np.minimum(dist_left, dist_right), 
                                  np.minimum(dist_bottom, dist_top))
    
    # Violation is how much r exceeds distance to boundary
    boundary_violation = np.maximum(0.0, r - dist_to_boundary)
    boundary_penalty = np.sum(boundary_violation**2)
    
    # 3. Overlap Penalties
    # Constraint: ||ci - cj|| >= r_i + r_j
    # Violation is how much (r_i + r_j) exceeds distance
    coords = np.column_stack((x, y))
    # Compute pairwise distance matrix
    # diff shape (n, n, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    radii_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    violation = radii_sum - dists
    violation = np.maximum(0.0, violation)
    
    # Sum of squared violations (divide by 2 to account for symmetry)
    overlap_penalty = np.sum(violation**2) / 2.0
    
    return obj + weight * (boundary_penalty + overlap_penalty)

def generate_initial_guess(seed, n):
    """
    Generates an initial guess based on a perturbed grid.
    """
    rng = np.random.RandomState(seed)
    
    # Create a grid of points (6x5 = 30 points)
    x_coords = np.linspace(0.1, 0.9, 6)
    y_coords = np.linspace(0.1, 0.9, 5)
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    all_x = grid_x.ravel()
    all_y = grid_y.ravel()
    
    # Select n points randomly to break symmetry
    indices = rng.choice(len(all_x), size=n, replace=False)
    cx = all_x[indices]
    cy = all_y[indices]
    
    # Add small random perturbation
    cx += rng.uniform(-0.02, 0.02, size=n)
    cy += rng.uniform(-0.02, 0.02, size=n)
    
    # Clip to valid range [0, 1]
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    
    # Initial radii
    r = np.ones(n) * 0.05
    
    return np.concatenate([cx, cy, r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to solve the circle packing problem.
    Returns centers (n, 2), radii (n,), and sum_radii.
    """
    n = N_CIRCLES
    weight = PENALTY_WEIGHT
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    best_params = None
    min_obj = np.inf
    num_trials = 20
    
    # Run optimization multiple times with different seeds
    for i in range(num_trials):
        x0 = generate_initial_guess(seed=i, n=n)
        
        try:
            # L-BFGS-B is suitable for bound-constrained optimization
            res = minimize(objective_function, x0, args=(n, weight), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < min_obj:
                min_obj = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is not None:
        x = best_params[:n]
        y = best_params[n:2*n]
        r = best_params[2*n:3*n]
        
        # Post-processing: Scale radii to ensure strict validity
        # This handles any tiny numerical violations from the optimizer
        
        coords = np.column_stack((x, y))
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        radii_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Calculate scaling factor k for pair constraints
        # We need k * (r_i + r_j) <= dist_ij
        ratio_pair = np.where(radii_sum > 1e-12, dists / radii_sum, np.inf)
        k_pair = np.min(ratio_pair)
        
        # Calculate scaling factor k for boundary constraints
        # We need k * r_i <= dist_to_boundary
        dists_to_boundary = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        ratio_bound = np.where(r > 1e-12, dists_to_boundary / r, np.inf)
        k_bound = np.min(ratio_bound)
        
        k = min(k_pair, k_bound, 1.0)
        
        # Safety floor
        if k < 1e-9:
            k = 1e-9
            
        final_radii = r * k
        final_centers = np.column_stack((x, y))
        
        best_total_radius = np.sum(final_radii)
        return final_centers, final_radii, best_total_radius
    else:
        # Fallback in case optimization fails completely
        centers = np.random.rand(n, 2)
        radii = np.ones(n) * 0.01
        return centers, radii, np.sum(radii)
