# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a3c1a30f) state=c0c4b0e7 sum of radii=1.300263 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def generate_initial_hex_packing(n, width=1.0, height=1.0):
    """
    Generates an initial set of centers for n circles in a hexagonal lattice.
    """
    # Approximate number of rows needed
    # In hex packing, density is high. 
    # Heuristic: rows ~ sqrt(n * sqrt(3) / 2) ?
    # Let's estimate row count based on area.
    # Area per circle in hex lattice ~ 2 * sqrt(3) * r^2.
    # 1.0 = n * 2 * sqrt(3) * r^2 => r = sqrt(1 / (2 * n * sqrt(3)))
    # Diameter d = 2r. Spacing horizontal = d, vertical = d * sqrt(3)/2.
    
    est_r = np.sqrt(1.0 / (2.0 * n * np.sqrt(3)))
    spacing = 2.0 * est_r
    
    rows = int(np.ceil(np.sqrt(n * np.sqrt(3) / 2.0)))
    centers = []
    
    # Try to fill rows
    count = 0
    r_idx = 0
    y = 0.1 # Start offset
    
    while count < n:
        # Determine how many circles in this row
        # Width available is 1.0. Spacing is spacing.
        # Max cols = floor(1.0 / spacing) + 1 ?
        # Let's just try to place as many as fit or need
        cols_needed = n - count
        max_cols = int(np.ceil(1.0 / spacing)) + 1
        current_cols = min(cols_needed, max_cols)
        
        # Adjust x positions to center the row
        row_width = (current_cols - 1) * spacing
        start_x = (1.0 - row_width) / 2.0
        
        # Hexagonal shift for odd/even rows
        if r_idx % 2 == 1:
            start_x += spacing / 2.0
            # If shifted row doesn't fit well, adjust cols or position
            # Simple fix: if start_x + row_width + spacing/2 > 1, reduce cols or shift back
            if start_x + row_width > 1.0:
                 start_x = 1.0 - row_width - 0.05 # Pull back
        
        for c in range(current_cols):
            x = start_x + c * spacing
            centers.append([x, y])
            count += 1
        
        y += spacing * np.sqrt(3) / 2.0
        r_idx += 1
        if y + 2*est_r > height: # Safety break
            break
            
    centers = np.array(centers)
    # Ensure we have exactly n
    if len(centers) < n:
        # Fill remaining with random if logic failed
        extra = np.random.rand(n - len(centers), 2)
        centers = np.vstack([centers, extra])
    elif len(centers) > n:
        centers = centers[:n]
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the packing optimization to maximize sum of radii for 26 circles.
    """
    n = 26
    np.random.seed(42) # For reproducibility
    
    # 1. Initialization
    centers = generate_initial_hex_packing(n)
    
    # Initial radius estimate. 
    # For n=26, equal circles r ~ 0.1. 
    # We want to maximize sum, so we start with a reasonable r and try to increase.
    # Or we can treat r as variable. Let's fix equal radius for simplicity and optimize positions,
    # then see if we can increase r.
    
    # Better approach: Variable radii optimization using forces.
    # Start with small radii, expand them.
    
    radii = np.full(n, 0.05) # Start small
    
    # Optimization parameters
    max_iter = 2000
    alpha = 0.1 # Step size / learning rate
    temperature = 1.0
    
    # We will try to maximize a common radius R, but allow individual variations 
    # to help convergence, or just enforce equality if it converges well.
    # Actually, for max sum of radii, equal radii is usually optimal.
    # Let's try to optimize a single global radius R.
    
    # However, constraint satisfaction is key.
    # Let's use a penalty method.
    
    best_sum_radii = 0.0
    best_centers = centers
    best_radii = radii
    
    # We perform a simulated annealing like process on the centers,
    # trying to satisfy constraints for a target radius R.
    # We search for the max R.
    
    low_r = 0.01
    high_r = 0.15 # Upper bound guess
    found_r = 0.0
    
    # Binary search for max R? 
    # Optimization is hard, so let's just run one long optimization that grows R.
    
    current_r = 0.05
    centers_opt = centers.copy()
    
    # Pre-calculate indices for pairwise checks
    i_indices, j_indices = np.triu_indices(n, k=1)
    
    for step in range(max_iter):
        # Gradually increase target radius
        target_r = current_r + 0.00005 
        # Cap growth
        if target_r > 0.12: 
            target_r = 0.12 # Limit reasonable radius
            
        # Compute forces/penalties for current configuration with radius target_r
        # Overlap penalty
        # dist >= 2 * target_r
        
        # Calculate distances
        # Vectorized calculation might be slow for large n, but n=26 is small.
        c_i = centers_opt[i_indices]
        c_j = centers_opt[j_indices]
        diffs = c_i - c_j
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        min_dist = 2.0 * target_r
        overlaps = min_dist - dists
        overlaps = np.maximum(0, overlaps) # Only penalize overlaps
        
        # Boundary penalties
        # x >= target_r, x <= 1 - target_r
        # y >= target_r, y <= 1 - target_r
        
        boundary_pen_x_lo = np.maximum(0, target_r - centers_opt[:, 0])
        boundary_pen_x_hi = np.maximum(0, centers_opt[:, 0] - (1.0 - target_r))
        boundary_pen_y_lo = np.maximum(0, target_r - centers_opt[:, 1])
        boundary_pen_y_hi = np.maximum(0, centers_opt[:, 1] - (1.0 - target_r))
        
        # Total energy/penalty
        # We want to minimize this.
        # If penalty is 0, we are valid for target_r.
        
        # Instead of minimizing energy, let's apply forces directly to positions.
        forces = np.zeros_like(centers_opt)
        
        # Pairwise repulsion
        # If overlap > 0, push apart.
        # Force magnitude proportional to overlap?
        # F = overlap
        # Direction = (c_j - c_i) / dist
        
        for k in range(len(i_indices)):
            if overlaps[k] > 0:
                i, j = i_indices[k], j_indices[k]
                if dists[k] > 1e-9:
                    vec = c_j - c_i # Vector from i to j? No, we are iterating pairs.
                    # Actually diffs was c_i - c_j.
                    # To push i away from j: direction (c_i - c_j).
                    # Wait, diffs = c_i - c_j.
                    # Push i in direction diffs (away from j).
                    # Push j in direction -diffs (away from i).
                    
                    dir_vec = diffs[k] / dists[k]
                    force_mag = overlaps[k] * 0.5 # Scale force
                    
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag
                else:
                    # Coincident, random push
                    forces[i] += np.random.rand(2) - 0.5
                    forces[j] -= np.random.rand(2) - 0.5

        # Boundary forces
        # Push back into square
        # If x < r, push right (+). If x > 1-r, push left (-).
        forces[:, 0] += boundary_pen_x_lo - boundary_pen_x_hi
        forces[:, 1] += boundary_pen_y_lo - boundary_pen_y_hi
        
        # Update positions
        # Learning rate decay
        lr = alpha / (1.0 + step * 0.001)
        
        centers_opt += forces * lr
        
        # Check if we successfully maintained validity (approx)
        # If total penalty is very low, we might try to increase target_r faster?
        # But for now, constant growth.
        
        # Clamp positions to [0, 1] strictly to avoid wild excursions, 
        # though boundary forces should handle it.
        centers_opt = np.clip(centers_opt, 0, 1)
        
        # Check validity for current target_r
        # Re-calculate max overlap
        max_overlap = np.max(overlaps) if len(overlaps) > 0 else 0
        max_bound_pen = max(
            np.max(boundary_pen_x_lo), np.max(boundary_pen_x_hi),
            np.max(boundary_pen_y_lo), np.max(boundary_pen_y_hi)
        )
        
        total_penalty = max_overlap + max_bound_pen
        
        if total_penalty < 1e-4:
            # Valid configuration for target_r
            # Save as best
            if target_r > found_r:
                found_r = target_r
                best_centers = centers_opt.copy()
                best_radii = np.full(n, target_r)
        else:
            # If penalty is high, maybe reduce target_r slightly or just let it be
            # The forces will try to fix it.
            pass
            
        # Decay temperature for random noise if we added any?
        # We didn't add noise, deterministic forces.
        
    # Final validation and cleanup
    # The radii should be set to the found_r
    # But wait, the problem allows different radii.
    # If we enforced equal radii, we found max equal radius.
    # Is it possible to have sum > 26 * found_r with unequal?
    # Maybe. But equal is a good baseline.
    # Let's try to optimize individual radii in a post-processing step?
    # Or just return the equal radius solution.
    # Given the target 2.636, and 26 * 0.1014 = 2.636, 
    # if found_r is around 0.101, we are good.
    
    # Let's re-run a quick optimization allowing variable radii to squeeze more sum.
    # Maximize sum(r) subject to constraints.
    # Use gradient ascent on radii?
    # Or just set radii = found_r * 1.05 and let positions adjust?
    
    # Let's stick to the equal radius result for robustness, 
    # but ensure we report the sum.
    
    # Re-verify constraints strictly
    # Recalculate valid radius based on best_centers
    # Find max r such that valid
    min_pair_dist = 1.0
    if len(i_indices) > 0:
        c_i = best_centers[i_indices]
        c_j = best_centers[j_indices]
        diffs = c_i - c_j
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_pair_dist = np.min(dists)
    
    max_r_inter = min_pair_dist / 2.0
    
    # Boundary constraints
    max_r_x = np.min(np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]))
    max_r_y = np.min(np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1]))
    max_r_bound = np.min([max_r_x, max_r_y])
    
    final_r = min(max_r_inter, max_r_bound)
    
    # Ensure non-negative
    if final_r < 0: final_r = 0
    
    final_radii = np.full(n, final_r)
    
    sum_radii = np.sum(final_radii)
    
    # Check if sum_radii is good.
    # If not, maybe the equal assumption was limiting?
    # But 2.636 is very specific. 2.636 / 26 = 0.10138.
    # If our found_r is close to 0.101, we are done.
    
    # Fallback: If sum is low, try a randomized local search for variable radii?
    # Too complex for this block.
    # The force method with growing radius usually finds the densest packing.
    
    return best_centers, final_radii, float(sum_radii)
