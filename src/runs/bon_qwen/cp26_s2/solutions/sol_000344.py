# sol_000344 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 776f37f0) state=4705e781 sum of radii=2.296809 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(params, n_circles, penalty_weight):
    """
    Computes the objective function for the packing problem.
    Objective: Maximize sum of radii (minimize negative sum).
    Penalizes overlaps and boundary violations.
    """
    # Unpack parameters: [x1, y1, r1, x2, y2, r2, ...]
    centers = params[:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    # 1. Boundary Penalties
    # Circles must be within [0, 1] x [0, 1].
    # Constraints: r <= x <= 1-r  and  r <= y <= 1-r.
    # Violation is positive if constraint is broken.
    
    # x < r  =>  r - x > 0
    v_left = np.maximum(0, radii - x)
    # x > 1-r  =>  x + r - 1 > 0
    v_right = np.maximum(0, x + radii - 1)
    # y < r  =>  r - y > 0
    v_bottom = np.maximum(0, radii - y)
    # y > 1-r  =>  y + r - 1 > 0
    v_top = np.maximum(0, y + radii - 1)
    
    boundary_penalty = np.sum(v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
    
    # 2. Overlap Penalties
    # Distance between centers i and j must be >= r_i + r_j.
    # Violation if r_i + r_j > distance.
    
    # Compute pairwise Euclidean distances
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Matrix of sum of radii
    R = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Violation amount
    violations = np.maximum(0, R - dist)
    # Ignore diagonal (distance to self is 0, sum of radii is 2r, but self-overlap is not a constraint)
    np.fill_diagonal(violations, 0)
    
    overlap_penalty = np.sum(violations**2)
    
    # Objective: minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Total loss with penalty
    return obj + penalty_weight * (overlap_penalty + boundary_penalty)

def run_packing():
    n = 26
    
    # Initialization Strategy: Hexagonal Packing Pattern
    # We initialize circles in a hexagonal lattice arrangement, which is the densest packing
    # for equal circles. This provides a strong starting point for the optimizer.
    # Layout: 5 rows with counts [6, 5, 6, 5, 4] = 26 circles.
    
    r_geom = 0.08  # Initial guess for radius
    dy = r_geom * np.sqrt(3)  # Vertical spacing in hexagonal packing
    
    # Calculate vertical offset to center the block of 5 rows in the unit square
    # Total height spanned by circles = 2*r (top/bottom) + 4 * dy (gaps between 5 rows)
    total_height = 2 * r_geom + 4 * dy
    y_offset = (1.0 - total_height) / 2.0
    
    centers_list = []
    row_counts = [6, 5, 6, 5, 4]
    current_y = y_offset + r_geom
    
    for row_idx, count in enumerate(row_counts):
        # Horizontal positioning: Center the row of 'count' circles in [0, 1]
        # Width occupied by centers = (count - 1) * 2 * r_geom
        width = (count - 1) * 2 * r_geom
        x_start = (1.0 - width) / 2.0
        
        for k in range(count):
            x = x_start + k * 2 * r_geom
            centers_list.append([x, current_y])
        
        current_y += dy
    
    centers = np.array(centers_list)
    radii = np.array([r_geom] * n)
    
    # Flatten parameters for the optimizer: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.hstack([centers.flatten(), radii])
    
    # Bounds for variables
    # x, y coordinates must be in [0, 1]
    # Radii must be non-negative and <= 0.5 (since diameter <= 1)
    bounds = []
    for _ in range(2 * n):
        bounds.append((0.0, 1.0))
    for _ in range(n):
        bounds.append((0.0, 0.5))
    
    penalty_weight = 10000.0
    
    # Run optimization using L-BFGS-B (supports bounds)
    # We maximize sum of radii by minimizing negative sum, with penalties for constraints.
    result = minimize(
        compute_loss,
        x0,
        args=(n, penalty_weight),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9}
    )
    
    best_params = result.x
    best_centers = best_params[:2*n].reshape(n, 2)
    best_radii = best_params[2*n:]
    
    # Post-processing: Validate and repair if numerical errors caused slight violations
    # Check pairwise overlaps
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    R = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
    violations = np.maximum(0, R - dist)
    np.fill_diagonal(violations, 0)
    max_viol_overlap = np.max(violations)
    
    # Check boundary violations
    x = best_centers[:, 0]
    y = best_centers[:, 1]
    r = best_radii
    
    # Max violation for each boundary type
    v_left = np.max(np.maximum(0, r - x))
    v_right = np.max(np.maximum(0, x + r - 1))
    v_bottom = np.max(np.maximum(0, r - y))
    v_top = np.max(np.maximum(0, y + r - 1))
    
    max_viol_boundary = max(v_left, v_right, v_bottom, v_top)
    
    max_viol = max(max_viol_overlap, max_viol_boundary)
    
    # If there are violations, shrink radii uniformly to satisfy constraints.
    # Shrinking by 'max_viol' ensures boundary constraints are met (r decreases by max_viol).
    # For overlaps, r_i + r_j decreases by 2*max_viol, clearing the violation.
    if max_viol > 1e-7:
        best_radii -= max_viol
        best_radii = np.maximum(best_radii, 0.0)
    
    # Ensure centers are strictly within bounds
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    sum_radii = float(np.sum(best_radii))
    
    return best_centers, best_radii, sum_radii
