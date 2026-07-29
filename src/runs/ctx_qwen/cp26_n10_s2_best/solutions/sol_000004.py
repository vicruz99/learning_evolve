# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=4e103e1d sum of radii=2.250000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.ndarray of shape (26, 2) with (x, y) coordinates.
        radii: np.ndarray of shape (26,) with radius of each circle.
        sum_radii: float, the sum of all radii.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run multiple attempts to find the best local optimum
    num_attempts = 10
    
    for attempt in range(num_attempts):
        # --- Initialization ---
        # Generate a hexagonal-like grid of 26 points
        centers = np.zeros((n, 2))
        idx = 0
        
        # Hexagonal grid parameters
        # We want to cover the [0,1]x[0,1] area.
        # Approximate spacing for 26 items.
        # Rows can have 5 or 6 items.
        # Let's try a pattern of rows with varying counts to approximate hex packing.
        # Pattern: 6, 5, 6, 5, 4 (Total 26)
        row_counts = [6, 5, 6, 5, 4]
        
        # Calculate vertical spacing based on 5 rows
        # To allow for shifting, we need enough vertical space.
        # Let's space rows evenly with some margin.
        row_y_positions = np.linspace(0.15, 0.85, len(row_counts))
        
        for r_idx, count in enumerate(row_counts):
            y = row_y_positions[r_idx]
            # Shift x for odd rows to create hexagonal staggering
            offset = (0.5 / count) if (r_idx % 2 == 1) else 0.0
            # Actually, simpler offset logic:
            # If we have 'count' circles, width is 1.
            # Center spacing = 1 / (count - 1) if count > 1?
            # No, we can place them anywhere. Let's space them evenly in [0, 1].
            # To maximize density, we pack them tight.
            # But for initialization, even spacing is good.
            
            # Generate x coordinates
            if count == 1:
                xs = [0.5]
            else:
                # Distribute in [0, 1]
                # If shifted, shift by half the step size?
                # Let's just create points in [0, 1]
                xs = np.linspace(0, 1, count)
                
                # Apply shift for hex pattern
                if r_idx % 2 == 1:
                    step = 1.0 / (count + 1) # slightly different logic for shift?
                    # Let's just shift by a small amount to break symmetry
                    xs = xs + 0.05 
                    # Keep within bounds
                    # xs = np.clip(xs, 0, 1) # Don't clip yet, let optimizer fix
                    # Better: scale to fit or just keep and let optimizer handle.
                    # Let's stick to a valid initialization inside [0,1].
                    # If shifted, we might go out.
                    # Let's construct valid grid first.
                    pass
            
            # Robust initialization: 
            # Just place them in a grid that fits, then perturb.
            # 6x5 grid = 30 points. Pick 26.
            pass

        # Fallback robust initialization: Random points from a grid
        # Grid 6x5
        grid_x = np.linspace(0.1, 0.9, 6)
        grid_y = np.linspace(0.1, 0.9, 5)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_points = np.column_stack([gx.ravel(), gy.ravel()])
        
        # Shuffle and pick 26
        np.random.shuffle(grid_points)
        centers = grid_points[:n].copy()
        
        # Add small random perturbation
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Initial radii: small enough to not overlap
        radii = np.ones(n) * 0.04
        
        # Flatten variables
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds: x,y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
        
        # --- Optimization ---
        
        def objective(v):
            c = v[:2*n].reshape(n, 2)
            r = v[2*n:]
            
            # Objective: Maximize sum(r) -> Minimize -sum(r)
            val = -np.sum(r)
            
            # Penalty weight
            W = 5000.0
            
            # Boundary penalties
            # Circle i: x in [r, 1-r], y in [r, 1-r]
            # Violations: r - x, x + r - 1, r - y, y + r - 1
            # We penalize max(0, violation)^2
            
            # Vectorized boundary penalty
            p_left = np.maximum(0.0, r - c[:, 0])
            p_right = np.maximum(0.0, c[:, 0] + r - 1.0)
            p_bottom = np.maximum(0.0, r - c[:, 1])
            p_top = np.maximum(0.0, c[:, 1] + r - 1.0)
            
            pen_bound = np.sum(p_left**2 + p_right**2 + p_bottom**2 + p_top**2)
            
            # Overlap penalties
            # Distance between i and j must be >= r_i + r_j
            # Violation: (r_i + r_j) - dist_ij
            # Penalty: max(0, violation)^2
            
            pen_overlap = 0.0
            # Efficient pairwise calculation
            # N=26 is small, but vectorizing helps
            # Compute all pairwise distances and radius sums
            # dist_matrix[i, j] = ||c_i - c_j||
            # radius_sum_matrix[i, j] = r_i + r_j
            
            # Using broadcasting
            # c is (n, 2)
            # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  -> (n, n, 2)
            # dist = np.sqrt(np.sum(diff**2, axis=2)) -> (n, n)
            
            # r is (n,)
            # r_sum = r[:, np.newaxis] + r[np.newaxis, :] -> (n, n)
            
            # However, we only need upper triangle.
            # But full matrix is easier to code and N is small.
            
            diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            
            r_sums = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Overlap violations
            violations = np.maximum(0.0, r_sums - dists)
            
            # Sum of squared violations (upper triangle to avoid double counting, though factor 2 cancels in gradient direction usually, but for cost function we care about magnitude)
            # Actually, sum over all pairs is fine, just scale W appropriately.
            # Or sum upper triangle.
            triu_idx = np.triu_indices(n, k=1)
            pen_overlap = np.sum(violations[triu_idx]**2)
            
            return val + W * (pen_bound + pen_overlap)

        try:
            # Use L-BFGS-B
            res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            current_centers = res.x[:2*n].reshape(n, 2)
            current_radii = res.x[2*n:]
            current_sum = np.sum(current_radii)
            
            # --- Validation ---
            # Strict check to ensure we return a valid packing
            is_valid = True
            
            # Check NaN
            if np.isnan(current_centers).any() or np.isnan(current_radii).any():
                is_valid = False
            else:
                # Check radii non-negative
                if np.any(current_radii < 0):
                    is_valid = False
                else:
                    # Check boundaries
                    # x - r >= -1e-12 => r - x <= 1e-12
                    # x + r <= 1 + 1e-12
                    for i in range(n):
                        x, y = current_centers[i]
                        r = current_radii[i]
                        if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # Check overlaps
                        for i in range(n):
                            for j in range(i + 1, n):
                                dist = np.sqrt(np.sum((current_centers[i] - current_centers[j]) ** 2))
                                if dist < current_radii[i] + current_radii[j] - 1e-9:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
            
            if is_valid:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
                    
        except Exception:
            pass
            
    # Fallback if no valid packing found (should not happen with good initialization)
    if best_centers is None:
        # Create a simple valid packing: small circles in grid
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        idx = 0
        # 5x5 grid
        for r_idx in range(5):
            for c_idx in range(5):
                if idx < n:
                    # Place at grid points
                    # Radius 0.09 fits easily
                    best_centers[idx] = [0.1 + c_idx * 0.2, 0.1 + r_idx * 0.2]
                    best_radii[idx] = 0.09
                    idx += 1
        best_sum = np.sum(best_radii)
        best_radii = best_radii
        best_centers = best_centers

    return best_centers, best_radii, best_sum
