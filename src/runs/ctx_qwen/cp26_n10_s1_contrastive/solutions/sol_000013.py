# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 866072f0) state=0aa3cf2a sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import random

def run_packing():
    n = 26
    
    # Helper to compute penalties and objective
    # We want to maximize sum(radii), so we minimize -sum(radii) + penalty
    def objective_and_gradient(x, centers, radii):
        # This is a bit complex to vectorize efficiently for gradient in one go,
        # so we will rely on numerical gradient or simple penalty evaluation.
        # For L-BFGS-B, providing gradient is faster but optional. 
        # Given 52 vars, numerical gradient might be slow but acceptable.
        # Let's just compute the value.
        
        # Reshape
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        # Ensure radii are non-negative (though bounds handle this)
        radii = np.maximum(radii, 0)
        
        sum_radii = np.sum(radii)
        
        # Boundary penalties
        # Constraint: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violation: if x < r, penalty.
        # We can enforce bounds [r, 1-r] on x,y but r is variable.
        # Instead, use penalty: max(0, r - x)^2 etc.
        penalty = 0.0
        
        # Boundary checks
        # x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        # Vectorized boundary penalty
        # centers shape (n, 2), radii shape (n)
        # x coords: centers[:, 0]
        
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        
        # Violations
        v1 = np.maximum(0, radii - x_coords) # x < r
        v2 = np.maximum(0, radii - (1 - x_coords)) # x > 1-r
        v3 = np.maximum(0, radii - y_coords) # y < r
        v4 = np.maximum(0, radii - (1 - y_coords)) # y > 1-r
        
        penalty += 100.0 * (np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2))
        
        # Overlap penalties
        # For all pairs i < j
        # dist >= r_i + r_j  => dist - (r_i + r_j) >= 0
        # Violation if dist < r_i + r_j
        
        # Compute all pairwise distances efficiently
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
        # dists = np.sqrt(np.sum(diff**2, axis=2)) # (n, n)
        # This creates a 26x26 matrix, cheap.
        
        # To avoid O(N^2) in Python loop, use vectorization
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # r_i + r_j matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap violation
        overlap = np.maximum(0, r_sum - dists)
        
        # Sum of squared violations. Use a large weight.
        # Sum over upper triangle to avoid double counting, though full sum is okay.
        penalty += 1000.0 * np.sum(np.triu(overlap, k=1)**2)
        
        return -sum_radii + penalty

    # We need to wrap this for scipy
    def objective_wrapper(x):
        return objective_and_gradient(x, None, None)

    # Bounds for x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
        bounds.extend([(0.0, 0.5)])             # r

    best_sum = -1.0
    best_x = None
    
    # Strategy 1: Hexagonal Lattice Initialization
    # Try to fit 26 circles in a hexagonal pattern
    # Estimated radius ~0.1
    # Rows with counts: 5, 5, 5, 5, 5, 1? Or 5, 5, 5, 5, 4, 2?
    # Let's just generate a dense hex grid and pick 26 points.
    
    init_configs = []
    
    # Config 1: Distorted Grid (5x5 + 1)
    # Start with 5x5 grid at 0.1 spacing, radii 0.05
    # Add one at center of a gap
    c1_centers = []
    r1 = 0.05
    for i in range(5):
        for j in range(5):
            c1_centers.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    # 26th circle at (0.2, 0.2) which is a gap
    c1_centers.append([0.2, 0.2])
    c1_centers = np.array(c1_centers)
    x0_1 = np.concatenate([c1_centers.flatten(), np.full(n, r1)])
    init_configs.append(x0_1)
    
    # Config 2: Hexagonal Packing
    # Spacing dx = 2r, dy = sqrt(3)r
    # Let's estimate r=0.095
    r_est = 0.095
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    c2_centers = []
    y = r_est
    row = 0
    while y <= 1 - r_est:
        x = r_est + (0.5 if row % 2 == 1 else 0.0) * dx
        while x <= 1 - r_est:
            if len(c2_centers) < n:
                c2_centers.append([x, y])
            x += dx
        y += dy
        row += 1
    
    # If we didn't get 26, fill randomly
    while len(c2_centers) < n:
        c2_centers.append([random.uniform(0, 1), random.uniform(0, 1)])
    
    c2_centers = np.array(c2_centers[:n])
    x0_2 = np.concatenate([c2_centers.flatten(), np.full(n, r_est)])
    init_configs.append(x0_2)
    
    # Config 3: Random
    np.random.seed(42)
    c3_centers = np.random.rand(n, 2)
    x0_3 = np.concatenate([c3_centers.flatten(), np.full(n, 0.01)])
    init_configs.append(x0_3)

    # Run optimization
    # Using L-BFGS-B with numerical gradient approximation
    # We might need to increase maxiter
    
    for i, x0 in enumerate(init_configs):
        try:
            res = opt.minimize(
                objective_wrapper, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            # Check result
            val = res.fun
            # Since objective is -sum + penalty, we want low value (negative sum)
            # But penalty should be 0.
            # Extract sum radii from variables
            r_opt = res.x[2*n:]
            current_sum = np.sum(r_opt)
            
            # Check validity roughly (penalty should be small)
            # If penalty is 0, val = -sum
            if val < -current_sum + 1e-5: 
                # Valid packing found (approx)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = res.x
        except Exception as e:
            print(f"Optimization failed for config {i}: {e}")

    if best_x is None:
        # Fallback to simple grid
        centers = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.2, 0.2]])
        radii = np.full(26, 0.01)
        return centers, radii, 0.0

    # Extract best solution
    centers_opt = best_x[:2*n].reshape(n, 2)
    radii_opt = best_x[2*n:]
    
    # Post-processing: Clean up overlaps numerically
    # If any overlaps exist due to numerical error, reduce radii slightly
    # Also ensure boundary constraints strictly
    
    # Simple iterative relaxation to ensure strict validity
    for _ in range(10):
        max_violation = 0
        for k in range(n):
            # Check boundaries
            cx, cy = centers_opt[k]
            r = radii_opt[k]
            
            # Boundary
            needed_r = min(cx, 1-cx, cy, 1-cy)
            if r > needed_r + 1e-12:
                radii_opt[k] = needed_r
            
            # Check overlaps with others
            for l in range(n):
                if k == l: continue
                dist = np.sqrt(np.sum((centers_opt[k] - centers_opt[l])**2))
                r_l = radii_opt[l]
                # Constraint: r_k + r_l <= dist
                # r_k <= dist - r_l
                max_r_k = dist - r_l
                if radii_opt[k] > max_r_k + 1e-12:
                    radii_opt[k] = max_r_k
        
        # Recalculate sum
        final_sum = np.sum(radii_opt)
        
        # If we reduced radii, the centers might be too loose. 
        # But for the purpose of returning a valid solution, this is safer.
        # However, reducing radii might drop sum below target if we were tight.
        # The optimizer should have kept penalty 0, so this loop is just safety.
        
    return centers_opt, radii_opt, np.sum(radii_opt)
