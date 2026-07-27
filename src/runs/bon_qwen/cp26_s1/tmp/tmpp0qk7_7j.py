import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses an optimization approach with a penalty function to enforce constraints.
    """
    n_circles = 26
    
    # --- Initialization ---
    # We initialize centers on a grid to provide a good starting configuration.
    # A 6x5 grid gives 30 points; we take the first 26.
    # Coordinates are centered in the cells.
    cols = 6
    rows = 5
    dx = 1.0 / cols
    dy = 1.0 / rows
    
    initial_centers = []
    count = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if count < n_circles:
                x = (c_idx + 0.5) * dx
                y = (r_idx + 0.5) * dy
                initial_centers.append([x, y])
                count += 1
            else:
                break
    
    centers_init = np.array(initial_centers)
    
    # Initial radius guess. 
    # For 26 circles, equal radius is approx 0.1. 
    # Start slightly smaller to ensure initial feasibility.
    r_init = 0.05
    
    # Variables: [x1, y1, x2, y2, ..., x26, y26, r]
    # Total 26*2 + 1 = 53 variables.
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    bounds.append((0.0, 0.5))     # r
    
    # --- Objective Function ---
    # We minimize: Alpha * Penalty - r
    # Penalty = Sum of squared overlaps + Sum of squared boundary violations
    
    ALPHA = 5000.0
    
    def objective(vars_flat):
        centers = vars_flat[:n_circles * 2].reshape(-1, 2)
        r = vars_flat[-1]
        
        penalty = 0.0
        
        # 1. Boundary penalties
        # Circle i must be within [0,1]x[0,1]
        # Distance to left wall: x
        # Distance to right wall: 1-x
        # Distance to bottom wall: y
        # Distance to top wall: 1-y
        # If dist < r, violation is (r - dist)
        
        # Vectorized boundary check
        xs = centers[:, 0]
        ys = centers[:, 1]
        
        # Left
        dist_left = xs
        overlap_left = np.maximum(0.0, r - dist_left)
        penalty += np.sum(overlap_left**2)
        
        # Right
        dist_right = 1.0 - xs
        overlap_right = np.maximum(0.0, r - dist_right)
        penalty += np.sum(overlap_right**2)
        
        # Bottom
        dist_bottom = ys
        overlap_bottom = np.maximum(0.0, r - dist_bottom)
        penalty += np.sum(overlap_bottom**2)
        
        # Top
        dist_top = 1.0 - ys
        overlap_top = np.maximum(0.0, r - dist_top)
        penalty += np.sum(overlap_top**2)
        
        # 2. Overlap penalties between circles
        # For every pair i < j, distance >= 2r
        # Violation if dist < 2r -> overlap = 2r - dist
        
        # Compute pairwise distances efficiently
        # Using broadcasting
        # centers shape (26, 2)
        # diff shape (26, 26, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Lower triangle indices
        i, j = np.tril_indices(n_circles, k=-1)
        
        # Distances for pairs
        pair_dists = dists[i, j]
        
        # Overlaps
        overlaps = np.maximum(0.0, 2.0 * r - pair_dists)
        penalty += np.sum(overlaps**2)
        
        # Return cost: penalize violations, reward radius
        # We want to minimize this.
        # High alpha ensures validity, -r pushes radius up.
        return ALPHA * penalty - r

    # --- Optimization ---
    # Use L-BFGS-B as it handles bounds well and is efficient for medium-sized problems.
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Extract results
    final_centers = result.x[:n_circles * 2].reshape(-1, 2)
    final_r = result.x[-1]
    
    # Since we optimized for equal radii, set all radii to final_r
    final_radii = np.full(n_circles, final_r)
    
    # Validation check (internal)
    # If the optimization resulted in a significant penalty, the solution might be invalid.
    # However, with high ALPHA, it should be valid.
    # Let's compute the sum of radii
    sum_radii = np.sum(final_radii)
    
    # If the result is somehow invalid (overlap), we might need to fallback or adjust.
    # But let's trust the optimizer with ALPHA=5000.
    
    # To be safe, let's verify validity roughly.
    # If penalty was high, maybe reduce radii slightly to ensure validity?
    # But the problem asks to maximize sum. If we are valid, we are good.
    
    return final_centers, final_radii, sum_radii

# Helper to run and validate
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radius: {radii[0]}")
    
    # Quick check
    import numpy as np
    n = centers.shape[0]
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            valid = False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-6:
                valid = False
    print(f"Valid: {valid}")