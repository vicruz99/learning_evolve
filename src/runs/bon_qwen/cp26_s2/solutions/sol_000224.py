# sol_000224 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=b15f758e sum of radii=1.987927 correctness=1.0
# stdout(first 200): Circle 24 at (0.08817515763525131, 0.9095924719351406) with radius 0.09291605597776621 is outside the unit square Circle 24 at (0.08817515763525131, 0.9095924719351406) with radius 0.09198689541798855
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_initial_hex_packing(n_circles):
    """
    Generate an initial configuration of circles using a hexagonal grid pattern.
    """
    # Estimate radius for initial packing to be safe (small)
    # We will grow them later.
    r_init = 0.04
    
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    # Hexagonal packing logic
    # Rows offset by half width
    row_idx = 0
    col_idx = 0
    
    # We want to fill the square. 
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2 * r
    # But since we start small, we can just place them in a grid and let optimizer fix it.
    # A dense random or grid init is fine.
    
    # Let's try to place them in rows
    # Estimate number of columns per row
    # 26 circles. 
    # Maybe 6 rows?
    
    rows = []
    temp_n = n_circles
    r_temp = 0.1 # heuristic for layout
    width_per_circle = 2 * r_temp
    cols = int(1.0 / width_per_circle) + 1
    if cols < 2: cols = 2
    
    # Distribute n_circles into rows
    # Try to make rows as equal as possible or hex-like
    # Simple distribution
    num_rows = int(np.ceil(n_circles / cols))
    
    current_y = 0
    # Adjust vertical step to fit in [0, 1]
    # We need num_rows rows.
    # y coordinates: r, r + step, ...
    # step = (1 - 2*r) / (num_rows - 1) ? No, we just place them, r will change.
    # Let's just place centers in a grid.
    
    # Better: Random initialization within bounds to avoid bad local minima from grid
    # But grid is safer for "no initial overlap".
    
    # Let's use a slightly randomized grid
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    x_coords = np.linspace(0.1, 0.9, grid_size)
    y_coords = np.linspace(0.1, 0.9, grid_size)
    
    pts = []
    for y in y_coords:
        for x in x_coords:
            pts.append([x, y])
            if len(pts) >= n_circles:
                break
        if len(pts) >= n_circles:
            break
            
    # Shuffle to randomize order? Maybe not needed.
    # But for 26, 5x5=25, need 1 more.
    # If grid is 6x6=36.
    
    # Let's just pick the first n_circles points
    selected = np.array(pts[:n_circles])
    
    # Add small jitter
    jitter = 0.02 * np.random.randn(*selected.shape)
    selected += jitter
    selected = np.clip(selected, 0.05, 0.95)
    
    return selected, np.full(n_circles, 0.02)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Try multiple restarts
    num_restarts = 5
    
    for _ in range(num_restarts):
        # Initialization
        centers, radii = get_initial_hex_packing(n_circles)
        
        # Optimization parameters
        lr_centers = 0.01
        lr_radii = 0.01
        penalty_lambda = 100.0
        max_iter = 2000
        
        # To speed up convergence, we can ramp up lambda
        # Or just keep it high.
        # High lambda enforces constraints strictly.
        
        for step in range(max_iter):
            # Compute gradients
            grad_c = np.zeros_like(centers)
            grad_r = np.zeros_like(radii)
            
            # Objective gradient: d(sum r)/dr = 1
            grad_r[:] = 1.0
            
            # Pairwise constraints
            # O_ij = r_i + r_j - dist_ij
            # Penalty term: - lambda * sum(O_ij^2) for O_ij > 0
            # Grad w.r.t X_i: + 2*lambda*O_ij * (X_i - X_j)/dist
            # Grad w.r.t R_i: - 2*lambda*O_ij
            
            # Vectorized pairwise distance calculation
            # centers shape (N, 2)
            # diffs shape (N, N, 2)
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diffs**2, axis=2)
            dists = np.sqrt(np.maximum(dists_sq, 1e-12)) # Avoid div by zero
            
            # Create mask for upper triangle i < j
            # We need to apply forces symmetrically
            # Force on i from j is F_ij
            # Force on j from i is -F_ij
            
            # Calculate overlaps
            radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlaps = radii_sum - dists
            
            # We only care about positive overlaps
            # Mask where i < j to avoid double counting in sum, 
            # but for gradients we need to distribute forces correctly.
            # Actually, easier to loop or use broadcasting carefully.
            
            # Let's compute force contributions
            # If overlap > 0:
            # F_mag = 2 * lambda * overlap
            # Direction = diffs / dists
            # grad_c[i] += lambda * 2 * overlap * (diffs[i,j] / dists[i,j])
            # Wait, deriv of - (r+r-d)^2 w.r.t x_i is + 2(r+r-d) * (x_i-x_j)/d
            
            # To avoid N^3 or heavy broadcasting, let's iterate or use sparse logic?
            # N=26 is small, N^2 is 676. Vectorized N^2 is fast.
            
            # Mask upper triangle
            mask = np.triu(np.ones((n_circles, n_circles)), k=1).astype(bool)
            
            # Filter pairs
            pairs_i = np.where(mask)[0]
            pairs_j = np.where(mask)[1]
            
            p_diffs = diffs[pairs_i, pairs_j] # (K, 2)
            p_dists = dists[pairs_i, pairs_j] # (K,)
            p_overlaps = overlaps[pairs_i, pairs_j] # (K,)
            
            # Only process active overlaps
            active = p_overlaps > 1e-7
            if np.any(active):
                a_diffs = p_diffs[active]
                a_dists = p_dists[active]
                a_overlaps = p_overlaps[active]
                a_i = pairs_i[active]
                a_j = pairs_j[active]
                
                # Force magnitude factor: 2 * lambda * overlap
                # Direction: diff / dist
                # Gradient contribution to centers: + factor * direction
                # Gradient contribution to radii: - 2 * lambda * overlap (for both i and j)
                
                # Wait, deriv w.r.t R_i is -2*lambda*overlap.
                # So for pair (i,j), R_i gets -2*lambda*overlap, R_j gets -2*lambda*overlap.
                
                factor = 2.0 * penalty_lambda * a_overlaps / a_dists # scalar per pair
                
                # Apply to centers
                # grad_c[i] += factor * a_diffs
                # grad_c[j] -= factor * a_diffs
                
                # Using np.add.at for accumulation
                np.add.at(grad_c, a_i, factor[:, np.newaxis] * a_diffs)
                np.add.at(grad_c, a_j, -factor[:, np.newaxis] * a_diffs)
                
                # Apply to radii
                np.add.at(grad_r, a_i, -2.0 * penalty_lambda * a_overlaps)
                np.add.at(grad_r, a_j, -2.0 * penalty_lambda * a_overlaps)
            
            # Boundary constraints
            # x=0: overlap = r - x. Force pushes x positive.
            # Grad x: + 2*lambda*(r-x). Grad r: -2*lambda*(r-x).
            # x=1: overlap = r - (1-x) = r + x - 1. Force pushes x negative.
            # Grad x: -2*lambda*(r+x-1). Grad r: -2*lambda*(r+x-1).
            
            # Left wall (x=0)
            ov_left = radii - centers[:, 0]
            active_left = ov_left > 1e-7
            if np.any(active_left):
                idx_l = np.where(active_left)[0]
                val_l = ov_left[idx_l]
                grad_c[idx_l, 0] += 2.0 * penalty_lambda * val_l
                grad_r[idx_l] -= 2.0 * penalty_lambda * val_l
            
            # Right wall (x=1)
            ov_right = radii + centers[:, 0] - 1.0
            active_right = ov_right > 1e-7
            if np.any(active_right):
                idx_r = np.where(active_right)[0]
                val_r = ov_right[idx_r]
                grad_c[idx_r, 0] -= 2.0 * penalty_lambda * val_r
                grad_r[idx_r] -= 2.0 * penalty_lambda * val_r
                
            # Bottom wall (y=0)
            ov_bot = radii - centers[:, 1]
            active_bot = ov_bot > 1e-7
            if np.any(active_bot):
                idx_b = np.where(active_bot)[0]
                val_b = ov_bot[idx_b]
                grad_c[idx_b, 1] += 2.0 * penalty_lambda * val_b
                grad_r[idx_b] -= 2.0 * penalty_lambda * val_b
                
            # Top wall (y=1)
            ov_top = radii + centers[:, 1] - 1.0
            active_top = ov_top > 1e-7
            if np.any(active_top):
                idx_t = np.where(active_top)[0]
                val_t = ov_top[idx_t]
                grad_c[idx_t, 1] -= 2.0 * penalty_lambda * val_t
                grad_r[idx_t] -= 2.0 * penalty_lambda * val_t
            
            # Update positions and radii
            # Gradient Ascent: var += lr * grad
            # But for penalty terms, gradients push towards feasibility.
            # We are maximizing Objective - Penalty.
            # So we follow gradient.
            
            centers += lr_centers * grad_c
            radii += lr_radii * grad_r
            
            # Clip values
            centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
            radii = np.clip(radii, 1e-6, 0.5) # Max radius 0.5
            
            # Annealing / Schedule
            # Decrease learning rate
            if step % 500 == 0:
                lr_centers *= 0.9
                lr_radii *= 0.9
            
            # Increase penalty to tighten constraints
            if step % 200 == 0 and penalty_lambda < 10000:
                penalty_lambda *= 1.1

        # Final validation and cleanup
        # Ensure radii are valid given centers (project onto feasible set)
        # This handles any remaining tiny overlaps or boundary issues
        # We can do a few steps of pure repulsion with fixed radii?
        # Or just clamp.
        
        # Check if valid
        if validate_packing(centers, radii):
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        else:
            # If invalid, try to fix by reducing radii slightly
            # Find min overlap and reduce
            # But for simplicity, just take what we have if sum is high?
            # The prompt requires valid packing.
            # Let's try a quick fix: reduce radii until valid.
            # Worst case, reduce all radii by small amount.
            scale = 0.99
            temp_r = radii * scale
            # Iteratively reduce until valid
            while not validate_packing(centers, temp_r):
                scale *= 0.95
                temp_r = radii * scale
                if scale < 0.01: break
            best_sum_temp = np.sum(temp_r)
            if best_sum_temp > best_sum:
                best_sum = best_sum_temp
                best_centers = centers.copy()
                best_radii = temp_r.copy()

    # If best is still None (unlikely), return a safe default
    if best_centers is None:
        best_centers = np.random.rand(26, 2) * 0.8 + 0.1
        best_radii = np.full(26, 0.05)
        best_sum = 26 * 0.05

    return best_centers, best_radii, float(best_sum)
