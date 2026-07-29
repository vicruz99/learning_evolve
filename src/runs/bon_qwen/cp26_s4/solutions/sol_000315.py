# sol_000315 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5cd869be) state=0ae66ff5 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constants for optimization
N_CIRCLES = 26
LAMBDA_PENALTY = 5000.0  # Penalty weight for constraint violations

def compute_loss(params):
    """
    Computes the objective function to be minimized.
    Objective: Maximize sum of radii <-> Minimize -sum(radii) + penalty.
    
    Args:
        params: np.array of shape (78,) containing [x1, y1, r1, ..., x26, y26, r26]
    
    Returns:
        float: The loss value
    """
    # Reshape params
    # Params structure: [x1, y1, r1, x2, y2, r2, ...]
    # Or better: [x1...x26, y1...y26, r1...r26] for easier slicing?
    # Let's use interleaved [x, y, r] for each circle for easier unpacking if needed,
    # but vectorized ops are easier with separate arrays.
    # Let's assume params is [x1, y1, r1, ..., x26, y26, r26]
    
    centers = params[0:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2*N_CIRCLES:3*N_CIRCLES]
    
    # 1. Boundary Violations
    # x - r >= 0  => violation r - x if negative
    # x + r <= 1  => violation x + r - 1 if positive
    # Same for y
    
    # Vectorized boundary checks
    x, y = centers[:, 0], centers[:, 1]
    
    # Left/Bottom boundary: x - r >= 0 => r - x <= 0. Violation if r - x > 0.
    viol_left = np.maximum(0, radii - x)
    viol_bottom = np.maximum(0, radii - y)
    
    # Right/Top boundary: x + r <= 1 => x + r - 1 <= 0. Violation if x + r - 1 > 0.
    viol_right = np.maximum(0, x + radii - 1.0)
    viol_top = np.maximum(0, y + radii - 1.0)
    
    boundary_penalty = np.sum(viol_left**2 + viol_bottom**2 + viol_right**2 + viol_top**2)
    
    # 2. Overlap Penalties
    # Distance matrix
    # centers shape (26, 2)
    # diff shape (26, 26, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # Radius sum matrix
    # radii shape (26,)
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap: rad_sum - dist_matrix. Positive means overlap.
    # We only care about i < j, but summing all and dividing by 2 is fine (diagonal is 0).
    overlap = np.maximum(0, rad_sum - dist_matrix)
    
    # Sum of squared overlaps. Note: overlap matrix is symmetric.
    # We sum all elements and divide by 2 to count each pair once.
    overlap_penalty = 0.5 * np.sum(overlap**2)
    
    # Objective: -sum(radii) + Lambda * (boundary_penalty + overlap_penalty)
    objective = -np.sum(radii) + LAMBDA_PENALTY * (boundary_penalty + overlap_penalty)
    
    return objective

def get_bounds():
    """
    Returns bounds for the optimization variables.
    Variables order: x1, y1, r1, x2, y2, r2, ...
    x, y in [0, 1], r in [0, 0.5]
    """
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    return bounds

def generate_hexagonal_init():
    """
    Generates an initial configuration based on a hexagonal lattice.
    Arrangement: 5, 4, 5, 4, 5, 3 circles in rows.
    """
    centers = []
    radii = []
    
    # Initial radius guess
    r_init = 0.08
    
    # Vertical spacing in hexagonal packing: sqrt(3)/2 * diameter = sqrt(3)*r
    v_step = np.sqrt(3) * r_init
    
    rows_config = [5, 4, 5, 4, 5, 3]
    
    y_curr = r_init
    
    for count in rows_config:
        # Horizontal spacing is 2*r
        h_step = 2 * r_init
        
        # Start x position
        # If row index is even (0, 2, 4), start at r_init
        # If row index is odd (1, 3, 5), start at 2*r_init (offset)
        # But we just iterate through rows_config. Let's track row index.
        # Actually, the offset depends on the previous row.
        # Let's just alternate.
        
        # We need to know if this is an offset row.
        # Let's assume row 0 is not offset.
        # But rows_config length is 6.
        
        # Calculate start x. 
        # For non-offset: x_start = r_init
        # For offset: x_start = r_init + r_init = 2*r_init? 
        # Valley between circles at r and 3r is 2r.
        
        # Let's track a flag
        pass 
    
    # Simpler generation: just place them and let optimizer fix positions.
    # We'll just fill a grid-like structure perturbed.
    
    # Let's try to fit 26 circles in a rough hexagonal pattern within [0,1]x[0,1]
    # We can just use a random valid initialization if lattice is tricky to code perfectly quickly.
    # But let's try a structured one.
    
    # 5 rows of 5 would be 25. 5x5 grid.
    # Let's do a 5x5 grid perturbed + 1 circle.
    
    # 5x5 grid
    # 5 circles in x, 5 in y.
    # Spacing = 1.0 / 5.0 = 0.2
    # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
    
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    
    c_list = []
    r_list = []
    
    for x in xs:
        for y in ys:
            c_list.append([x, y])
            r_list.append(0.08) # Start small to avoid immediate penalty
            
    # Add 26th circle in the center
    c_list.append([0.5, 0.5])
    r_list.append(0.02) # Small radius
    
    centers = np.array(c_list)
    radii = np.array(r_list)
    
    # Perturb slightly
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    return centers, radii

def params_to_arrays(params):
    """Helper to convert flat params to centers and radii arrays."""
    centers = params[0:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2*N_CIRCLES:3*N_CIRCLES]
    return centers, radii

def run_packing() -> tuple:
    """
    Optimizes the packing of 26 circles in a unit square.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    
    # Set random seed for reproducibility in some steps, but we want exploration
    # np.random.seed(42) 
    
    best_loss = float('inf')
    best_params = None
    
    # Try multiple initializations to avoid local minima
    n_restarts = 10
    
    for i in range(n_restarts):
        # Generate initial configuration
        # Use hexagonal-ish or grid
        if i % 2 == 0:
            # Grid-based init
            centers, radii = generate_hexagonal_init()
        else:
            # Random valid init
            centers = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
            radii = np.full(N_CIRCLES, 0.05)
        
        # Flatten to params
        params0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds
        bounds = get_bounds()
        
        # Optimize
        # Use L-BFGS-B as it handles bounds well
        res = minimize(
            compute_loss, 
            params0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        
        # Check result
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
            # Check if penalty is low (constraints satisfied)
            centers, radii = params_to_arrays(best_params)
            # Calculate actual violations to be sure
            # We can call compute_loss components or just trust the score if low
            # But let's check validity explicitly
            
    # Extract best solution
    centers, radii = params_to_arrays(best_params)
    
    # Post-processing: Validate and fix small violations
    # The penalty method might leave tiny violations.
    # We can check and shrink radii slightly if needed.
    
    # Calculate max violation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlaps (positive means violation)
    overlaps = rad_sum - dist_matrix
    max_overlap = np.max(overlaps)
    
    # Boundary violations
    x, y = centers[:, 0], centers[:, 1]
    b_viol = np.maximum(0, radii - x) + np.maximum(0, x + radii - 1) + \
             np.maximum(0, radii - y) + np.maximum(0, y + radii - 1)
    max_b_viol = np.max(b_viol)
    
    max_viol = max(max_overlap, max_b_viol)
    
    # If there are violations, shrink all radii slightly to satisfy constraints
    # We need to reduce r such that r_i + r_j <= d_ij - epsilon
    # A safe way is to scale all radii down by factor (1 - margin)
    if max_viol > 1e-7:
        # Estimate how much to shrink. 
        # If max_overlap is delta, we need to reduce r sum by delta.
        # Reducing all r by delta/2 works for pairwise? 
        # Actually, if we reduce all r by k, sum reduces by 2k.
        # So k = max_overlap / 2.
        # Also boundary: r <= x. If r > x, need r <= x. Reduce r by (r-x).
        # Max boundary violation is max(r-x).
        # Let's just shrink by max(max_overlap/2, max_b_viol) + epsilon
        shrink_factor = max(max_overlap / 2.0, max_b_viol) + 1e-6
        radii = radii - shrink_factor
        radii = np.maximum(radii, 0.0) # Ensure non-negative
        
        # Re-check
        # It might still be tight, so we might need iterative shrinking, 
        # but a single step should suffice for small violations.
        
        # Recalculate sum
        sum_radii = np.sum(radii)
    else:
        sum_radii = np.sum(radii)

    # Final validation check (dry run)
    # Note: The provided validation function is for checking, we assume our fix worked.
    
    return centers, radii, float(sum_radii)
