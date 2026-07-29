# sol_000135 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state da2150ba) state=e708bcd0 sum of radii=2.311896 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a penalty-based gradient descent method.
    """
    np.random.seed(42)
    n_circles = 26
    
    # --- Initialization ---
    # Initialize centers in a hexagonal-like grid pattern inside the square.
    # This provides a good starting distribution.
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Approximate grid parameters
    # Try to fit 26 points. 5 rows of 5 is 25, plus 1 extra.
    # Let's use a staggered grid.
    # Rows y-coords
    num_rows = 6
    y_coords = np.linspace(0.1, 0.9, num_rows)
    
    # Distribute circles into rows
    # We want roughly equal number per row, staggering them.
    # 26 circles. 6 rows. 4, 5, 4, 5, 4, 4 ? Sum = 26.
    # Or 5, 5, 5, 5, 4, 2?
    # Let's try to fill rows 6, 5, 5, 5, 4, 1?
    # Let's just place them based on index to fill the square well.
    
    idx = 0
    row_counts = [5, 5, 5, 5, 4, 2] # Sum = 26
    # Adjust to make it more uniform? 
    # 5, 5, 5, 5, 4, 2 is okay.
    
    current_row = 0
    for count in row_counts:
        # Stagger x positions
        # Even rows (0-indexed) aligned, odd rows shifted
        x_start = 0.05 if current_row % 2 == 1 else 0.05
        # Spacing
        step = (1.0 - 0.1) / count # Fill width roughly
        
        # Generate centers
        for i in range(count):
            if idx >= n_circles:
                break
            x = x_start + i * step
            y = y_coords[current_row]
            centers[idx] = [x, y]
            radii[idx] = 0.05 # Initial small radius
            idx += 1
        
        current_row += 1
    
    # Fill any remaining if logic was off (should be exact with sum=26)
    while idx < n_circles:
        centers[idx] = [np.random.rand(), np.random.rand()]
        radii[idx] = 0.05
        idx += 1

    # --- Optimization Parameters ---
    penalty_weight = 500.0  # Weight for overlap/boundary penalties
    lr = 0.005              # Learning rate
    num_iterations = 2000   # Number of optimization steps
    
    # Ensure centers are within bounds initially (just in case)
    centers = np.clip(centers, 0, 1)

    for step in range(num_iterations):
        # Compute forces/gradients
        grad_centers = np.zeros_like(centers)
        grad_radii = np.zeros_like(radii)
        
        # 1. Gradient from objective: maximize sum(r) => minimize -sum(r)
        # Gradient of -sum(r) w.r.t r_i is -1
        grad_radii[:] = -1.0
        
        # 2. Overlap Penalties
        # For each pair (i, j)
        # Vectorized distance calculation
        # centers shape (N, 2)
        # diff shape (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-12) # Avoid div by zero
        
        # Radii sum matrix (N, N)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: max(0, r_i + r_j - dist)
        # We only care about upper triangle to avoid double counting, 
        # but for gradient accumulation on each circle, we can iterate all or handle symmetry.
        # Let's iterate pairs to be safe and clear.
        
        # Create a mask for i < j
        i_indices, j_indices = np.triu_indices(n_circles, k=1)
        
        # Calculate distances and overlaps for pairs
        d_ij = dist[i_indices, j_indices]
        r_sum_ij = r_sum[i_indices, j_indices]
        overlaps = r_sum_ij - d_ij
        
        # Only consider positive overlaps
        mask = overlaps > 0
        active_overlaps = overlaps[mask]
        active_d = d_ij[mask]
        active_i = i_indices[mask]
        active_j = j_indices[mask]
        
        if len(active_i) > 0:
            # Gradient contribution:
            # Penalty term: 0.5 * C * (overlap)^2
            # d(Penalty)/d(x_i) = C * overlap * d(overlap)/d(x_i)
            # overlap = r_i + r_j - dist
            # d(overlap)/d(x_i) = - d(dist)/d(x_i) = - (x_i - x_j) / dist
            # So gradient on x_i is -C * overlap * (x_i - x_j) / dist
            # Wait, minimization moves opposite to gradient.
            # Force to push apart: + C * overlap * (x_i - x_j) / dist
            
            # Direction vector from j to i: (x_i - x_j)
            vec_ij = diff[active_i, active_j, :] # (k, 2)
            unit_vec_ij = vec_ij / active_d[:, np.newaxis] # Normalize
            
            # Magnitude of force
            force_mag = penalty_weight * active_overlaps # C * overlap
            
            # Gradient for centers: we want to move i away from j.
            # Gradient of cost is negative of force direction?
            # Cost increases with overlap. We want to decrease cost.
            # dCost/dx_i = - Force_mag * unit_vec_ij ?
            # Let's check: if overlap > 0, cost > 0.
            # Moving i away from j increases dist, decreases overlap, decreases cost.
            # Vector i->j is -unit_vec_ij. Moving in direction i->j decreases cost.
            # So gradient points towards j.
            # gradient = force_mag * (-unit_vec_ij) ? No.
            # Let's stick to math:
            # C(x) = 0.5 * C * (r_i + r_j - ||x_i - x_j||)^2
            # dC/dx_i = C * (r_i + r_j - d) * (- (x_i - x_j)/d)
            # dC/dx_i = - C * overlap * (x_i - x_j)/d
            # This is the gradient.
            # Update: x_i -= lr * grad.
            # x_i -= lr * (- C * overlap * unit_vec) => x_i += lr * C * overlap * unit_vec.
            # unit_vec = (x_i - x_j)/d points from j to i.
            # So x_i moves away from j. Correct.
            
            grad_centers[active_i] += -force_mag[:, np.newaxis] * unit_vec_ij
            grad_centers[active_j] += -force_mag[:, np.newaxis] * (-unit_vec_ij) # Symmetric
            
            # Gradient for radii
            # dC/dr_i = C * overlap * 1
            grad_radii[active_i] += force_mag
            grad_radii[active_j] += force_mag

        # 3. Boundary Penalties
        # Constraints: x >= r, 1-x >= r => x <= 1-r
        # Violation 1: x - r < 0 => r - x > 0. Penalty 0.5*C*(r-x)^2
        # Violation 2: (1-x) - r < 0 => x + r - 1 > 0. Penalty 0.5*C*(x+r-1)^2
        
        # Left boundary: x < r
        left_viol = radii - centers[:, 0]
        left_mask = left_viol > 0
        if np.any(left_mask):
            # Gradient d/dr: C*(r-x)
            # Gradient d/dx: -C*(r-x)
            grad_radii[left_mask] += penalty_weight * left_viol[left_mask]
            grad_centers[left_mask, 0] += -penalty_weight * left_viol[left_mask]
            
        # Right boundary: x + r > 1
        right_viol = centers[:, 0] + radii - 1.0
        right_mask = right_viol > 0
        if np.any(right_mask):
            # Gradient d/dr: C*(x+r-1)
            # Gradient d/dx: C*(x+r-1)
            grad_radii[right_mask] += penalty_weight * right_viol[right_mask]
            grad_centers[right_mask, 0] += penalty_weight * right_viol[right_mask]

        # Bottom boundary: y < r
        bottom_viol = radii - centers[:, 1]
        bottom_mask = bottom_viol > 0
        if np.any(bottom_mask):
            grad_radii[bottom_mask] += penalty_weight * bottom_viol[bottom_mask]
            grad_centers[bottom_mask, 1] += -penalty_weight * bottom_viol[bottom_mask]

        # Top boundary: y + r > 1
        top_viol = centers[:, 1] + radii - 1.0
        top_mask = top_viol > 0
        if np.any(top_mask):
            grad_radii[top_mask] += penalty_weight * top_viol[top_mask]
            grad_centers[top_mask, 1] += penalty_weight * top_viol[top_mask]

        # --- Update Parameters ---
        # Center update
        centers -= lr * grad_centers
        # Clip centers to [0, 1] to prevent wild excursions, though penalty handles it
        centers = np.clip(centers, 0.0, 1.0)
        
        # Radius update
        radii -= lr * grad_radii
        # Radii must be non-negative
        radii = np.maximum(radii, 0.001) # Keep small epsilon
        
        # Optional: Increase penalty weight slowly?
        # penalty_weight *= 1.001

    # Final cleanup to ensure strict validity (clip radii if touching boundaries slightly)
    # If a circle is slightly out, shrink it
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < r:
            radii[i] = max(0.0, max_r)
            
    # Re-check overlaps and shrink if necessary (post-processing)
    # Simple iterative shrink for overlaps
    for _ in range(100):
        changed = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j]:
                    # Overlap. Reduce radii.
                    # Reduce equally or proportional?
                    overlap = (radii[i] + radii[j]) - dist
                    radii[i] -= overlap / 2
                    radii[j] -= overlap / 2
                    changed = True
        if not changed:
            break
            
    # Re-clip radii for boundary
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1-x, y, 1-y)
        if max_r < r:
            radii[i] = max(0.0, max_r)

    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)
