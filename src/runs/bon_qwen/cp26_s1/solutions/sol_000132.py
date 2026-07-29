# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=8669e9dd sum of radii=0.001625 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def compute_loss(params, n, penalty):
    """
    Computes the objective function value: -sum(radii) + penalty * constraint_violations.
    
    Args:
        params: Flattened array of centers and radii.
        n: Number of circles.
        penalty: Weight for constraint violations.
    
    Returns:
        Objective value.
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    loss = 0.0
    
    # 1. Pairwise overlap violations
    # Compute pairwise distances
    # Shape (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Sum of radii for each pair
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount (positive if circles overlap)
    # We use a small epsilon to ignore numerical noise, but max(0, ...) handles logic
    overlap = rad_sum - dist
    overlap = np.maximum(overlap, 0.0)
    # Sum of squared overlaps
    loss += np.sum(overlap**2)
    
    # 2. Boundary violations
    # Circle i with center (x, y) and radius r must satisfy:
    # r <= x <= 1-r  <=>  x - r >= 0 and 1 - x - r >= 0
    # r <= y <= 1-r  <=>  y - r >= 0 and 1 - y - r >= 0
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    # Left wall: x < r  => violation r - x
    loss += np.sum(np.maximum(r - x, 0.0)**2)
    # Right wall: x > 1-r  => violation x + r - 1
    loss += np.sum(np.maximum(x + r - 1.0, 0.0)**2)
    # Bottom wall: y < r  => violation r - y
    loss += np.sum(np.maximum(r - y, 0.0)**2)
    # Top wall: y > 1-r  => violation y + r - 1
    loss += np.sum(np.maximum(y + r - 1.0, 0.0)**2)
    
    return loss

def objective(params, n, penalty):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize -sum(radii).
    We add a penalty term for constraint violations.
    """
    loss_val = compute_loss(params, n, penalty)
    radii = params[2*n:]
    return -np.sum(radii) + penalty * loss_val

def run_packing():
    n = 26
    
    # --- Initialization: Hexagonal Lattice ---
    # This provides a dense starting configuration
    centers = []
    r_temp = 0.12  # Temporary radius for layout
    y = r_temp
    x = r_temp
    row = 0
    
    # Generate enough points to fill the square
    while len(centers) < n + 10:
        x = r_temp
        # Offset odd rows for hexagonal packing
        if row % 2 == 1:
            x = r_temp + r_temp 
        
        while x <= 1.0 - r_temp and len(centers) < n + 10:
            centers.append([x, y])
            x += 2 * r_temp
        
        y += r_temp * math.sqrt(3) / 2
        row += 1
    
    init_centers = np.array(centers[:n])
    # Start with a small valid radius to ensure initial validity
    init_radii = np.full(n, 0.05) 
    
    # --- Optimization ---
    # We will use a penalty method. 
    # A high penalty forces the optimizer to respect constraints.
    penalty = 2000.0
    
    # Bounds for variables:
    # Centers: [0, 1] for both x and y
    # Radii: [0, 0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    best_obj = np.inf
    best_params = None
    best_centers = None
    best_radii = None
    
    # Run multiple restarts with random perturbations to find a good local minimum
    np.random.seed(42)
    
    for i in range(15):
        # Prepare initial parameters
        if i == 0:
            x0 = np.concatenate([init_centers.flatten(), init_radii])
        else:
            # Perturb the best found solution or initial one
            if best_params is not None:
                x0 = best_params.copy()
            else:
                x0 = np.concatenate([init_centers.flatten(), init_radii])
            
            # Add noise
            noise = np.random.normal(0, 0.01, size=x0.shape)
            x0 = x0 + noise
            
            # Clip to rough bounds to keep it valid-ish
            x0[:2*n] = np.clip(x0[:2*n], 0, 1)
            x0[2*n:] = np.clip(x0[2*n:], 0.01, 0.5)
            
        # Run optimizer
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(n, penalty),
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue

    # Extract results
    if best_params is None:
        # Fallback to initial if optimization failed completely
        best_centers = init_centers
        best_radii = init_radii
    else:
        best_centers = best_params[:2*n].reshape(n, 2)
        best_radii = best_params[2*n:]
        
    # Final validation and safety clipping
    # If the optimizer is on the boundary of validity, slight numerical errors might make it invalid.
    # We can perform a simple shrink if necessary, but usually the penalty method handles it.
    # We will return the computed best.
    
    return best_centers, best_radii, np.sum(best_radii)
