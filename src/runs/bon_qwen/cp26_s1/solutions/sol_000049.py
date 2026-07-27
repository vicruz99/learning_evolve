# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b794a107) state=f45ae5ff sum of radii=0.260000 correctness=1.0
# stdout(first 200): Optimization error: index 13 is out of bounds for axis 0 with size 13 Optimization error: index 13 is out of bounds for axis 0 with size 13 Optimization error: index 13 is out of bounds for axis 0 wit
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Function to generate an initial hexagonal-like configuration
    def get_initial_guess(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Estimate radius based on area and density (approx 0.9)
        # 26 * pi * r^2 * 0.9 <= 1  => r approx 0.105
        # But to be safe for initialization, start smaller
        r_init = 0.08
        
        centers = np.zeros((n_circles, 2))
        radii = np.ones(n_circles) * r_init
        
        # Hexagonal packing generation
        row_height = np.sqrt(3) * r_init
        col_width = 2 * r_init
        
        y_pos = r_init
        idx = 0
        row = 0
        
        while idx < n_circles:
            # Calculate max x for this row
            # Even rows: start at r_init
            # Odd rows: start at 2*r_init (offset)
            if row % 2 == 0:
                x_start = r_init
                # How many fit?
                # (n-1)*2r + 2r <= 1 => n <= 1/(2r)
                # Actually centers: x_i = r + i*2r. Last center 1-r.
                # Max i such that r + i*2r <= 1-r => i*2r <= 1-2r => i <= (1-2r)/2r = 1/(2r) - 1
                # Number of circles = i + 1 = 1/(2r)
                max_circles_in_row = int(np.floor(1 / (2 * r_init)))
            else:
                x_start = 2 * r_init
                # Shifted row: first center at 2r. Last at 1-r?
                # x_i = 2r + i*2r. Last x <= 1-r.
                # 2r + i*2r <= 1-r => i*2r <= 1-3r => i <= (1-3r)/2r
                # Number of circles = i + 1 = (1-3r)/2r + 1 = (1-r)/2r
                if r_init > 1/3: # Should not happen with r=0.08
                    max_circles_in_row = 0
                else:
                    max_circles_in_row = int(np.floor((1 - r_init) / (2 * r_init)))

            # Place circles in this row
            count = 0
            while count < max_circles_in_row and idx < n_circles:
                centers[idx, 0] = x_start + count * col_width
                centers[idx, 1] = y_pos
                idx += 1
                count += 1
            
            y_pos += row_height
            row += 1
            
        # Add small random noise to break symmetry and help optimization
        noise_scale = 0.005
        centers += np.random.uniform(-noise_scale, noise_scale, centers.shape)
        
        # Clip centers to valid range [r, 1-r] roughly, but let optimizer handle bounds
        # Just ensure they are inside [0,1]
        centers = np.clip(centers, 0.0, 1.0)
        
        return centers, radii

    # Objective function to minimize: -(Sum of Radii) + Penalties
    def objective(x):
        # x is flattened array: [x0, y0, r0, x1, y1, r1, ...]
        centers = x[0::3].reshape(-1, 2)
        radii = x[2::3]
        
        # Objective: Maximize sum of radii => Minimize negative sum
        obj_val = -np.sum(radii)
        
        # Penalty for overlaps
        # dist >= r_i + r_j
        # Penalty = max(0, r_i + r_j - dist)^2
        overlap_penalty = 0.0
        n = len(radii)
        # Vectorized distance calculation for performance
        # Expand dimensions: (n, 1, 2) and (1, n, 2)
        c_diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(c_diff**2, axis=2))
        
        # Lower triangle indices
        i, j = np.tril_indices(n, -1)
        
        # Overlaps
        overlaps = radii[i] + radii[j] - dists[i, j]
        # Only penalize positive overlaps
        positive_overlaps = np.maximum(0, overlaps)
        overlap_penalty += 100.0 * np.sum(positive_overlaps**2)
        
        # Penalty for boundary constraints
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # Penalty = sum of max(0, r - x)^2 etc.
        boundary_penalty = 0.0
        
        # x bounds
        boundary_penalty += 100.0 * np.sum(np.maximum(0, radii - centers[:, 0])**2)
        boundary_penalty += 100.0 * np.sum(np.maximum(0, radii - (1 - centers[:, 0]))**2)
        # y bounds
        boundary_penalty += 100.0 * np.sum(np.maximum(0, radii - centers[:, 1])**2)
        boundary_penalty += 100.0 * np.sum(np.maximum(0, radii - (1 - centers[:, 1]))**2)
        
        # Penalty for negative radii (though bounds should handle this)
        boundary_penalty += 100.0 * np.sum(np.maximum(0, -radii)**2)
        
        return obj_val + overlap_penalty + boundary_penalty

    def gradient(x):
        # Numerical gradient if analytic is too complex, but L-BFGS-B can approximate.
        # However, providing an analytical gradient or using a method that approximates it is better.
        # L-BFGS-B approximates gradient if not provided.
        return None 

    # We will use 'L-BFGS-B' as it supports bounds.
    # Variables: x_i, y_i, r_i
    # Bounds:
    # x: [0, 1]
    # y: [0, 1]
    # r: [0, 0.5] (max possible radius in unit square)
    
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    best_result = None
    
    # Try multiple random restarts to find global optimum
    for trial in range(10):
        # Generate initial guess
        # Use a slightly larger radius for initial guess to bias towards larger circles
        # But we need a valid starting point.
        # Let's use the hexagonal generator with r=0.09
        c_init, r_init = get_initial_guess(seed=trial * 123 + 42)
        
        # Flatten to 1D array
        x0 = np.zeros(n_circles * 3)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        # Optimize
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            if res.success or (res.nit == 2000):
                current_val = -res.fun # Remove penalty? No, fun includes penalty.
                # Actually, we need to extract the true sum of radii from the result
                r_sol = res.x[2::3]
                true_sum = np.sum(r_sol)
                
                # Check validity roughly
                # If penalty is low, solution is valid
                penalty_val = res.fun - (-true_sum) # fun = -sum + penalty => penalty = fun + sum
                
                if penalty_val < 1e-4 and true_sum > best_sum_radii:
                    best_sum_radii = true_sum
                    best_centers = res.x[0::3].reshape(-1, 2)
                    best_radii = r_sol
        except Exception as e:
            print(f"Optimization error: {e}")
            continue

    # If optimization failed to find a valid packing or we need to refine
    # Let's ensure validity and potentially adjust slightly
    if best_centers is not None:
        # Check for any remaining tiny violations and shrink if necessary
        # This is a safeguard
        c = best_centers
        r = best_radii
        
        # Check overlaps
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                needed = r[i] + r[j]
                if dist < needed - 1e-9:
                    # Overlap, shrink both slightly
                    shrink = (needed - dist) / 2 + 1e-6
                    r[i] -= shrink
                    r[j] -= shrink
        
        # Check boundaries
        for i in range(n_circles):
            x, y = c[i]
            ri = r[i]
            # Max allowed radius at this position
            max_r = min(x, 1-x, y, 1-y)
            if ri > max_r + 1e-9:
                r[i] = max_r
    
        best_sum_radii = np.sum(best_radii)
        
        # Final validation check (mental check)
        # If radii are negative, fix
        best_radii = np.maximum(best_radii, 0.0)
        
    else:
        # Fallback to a simple grid if optimizer fails completely
        # 5x5 grid + 1 small circle in center?
        # 5x5 grid r=0.1. 25 circles.
        # Add 26th?
        # Let's create a 5x5 grid of r=0.095 and add one?
        # Or just a dense random packing?
        # Fallback: 26 circles in grid
        # 6 rows: 5, 5, 5, 5, 4, 2?
        # Let's just use the optimizer result or a valid default.
        # Default valid packing:
        # 6 rows of circles.
        # y = 0.083, 0.25, 0.416, 0.583, 0.75, 0.916 (spacing 1/7 approx?)
        # Actually, just use the first valid result from the loop if any.
        # If not, construct a safe one.
        
        # Safe fallback: 26 circles of radius 0.01
        best_centers = np.random.rand(n_circles, 2)
        best_radii = np.full(n_circles, 0.01)
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
