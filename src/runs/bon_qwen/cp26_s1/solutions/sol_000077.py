# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22281c24) state=05e42789 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n_circles = 26
    
    def get_loss(optimized_vars):
        # Unpack variables
        # Structure: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # Total length: 26 * 3 = 78
        centers = optimized_vars[:52].reshape(26, 2)
        radii = optimized_vars[52:]
        
        penalty = 0.0
        target_sum = -np.sum(radii) # We maximize sum, so minimize negative sum
        
        # Boundary constraints and negative radii
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Penalty for radius < 0 (shouldn't happen with bounds, but safe to check)
            if r < 0:
                penalty += 1000.0 * abs(r)
            
            # Penalty for being outside [0,1]
            if x - r < 0: penalty += 1000.0 * abs(x - r)
            if x + r > 1: penalty += 1000.0 * abs(x + r - 1)
            if y - r < 0: penalty += 1000.0 * abs(y - r)
            if y + r > 1: penalty += 1000.0 * abs(y + r - 1)

        # Overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                dist = math.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    penalty += 1000.0 * (min_dist - dist)
        
        return target_sum + penalty

    # Best solution found
    best_solution = None
    best_score = -np.inf

    # Run multiple times with different initializations
    for _ in range(10):
        # 1. Initialize with a 5x5 grid for 25 circles, 1 circle in a gap
        centers_init = np.zeros((26, 2))
        radii_init = np.full(26, 0.1)
        
        grid_pts = [0.1, 0.3, 0.5, 0.7, 0.9]
        idx = 0
        for y in grid_pts:
            for x in grid_pts:
                if idx < 25:
                    centers_init[idx] = [x, y]
                    idx += 1
        
        # Place 26th circle in a gap (e.g., at 0.2, 0.2 with slightly smaller radius to avoid overlap initially)
        centers_init[25] = [0.2, 0.2]
        radii_init[25] = 0.04 

        # Random perturbation to help optimization
        centers_init += np.random.normal(0, 0.005, centers_init.shape)
        radii_init += np.random.normal(0, 0.005, radii_init.shape)
        radii_init = np.maximum(radii_init, 0.01) # Ensure positive radii

        # Flatten for optimizer
        x0 = np.concatenate([centers_init.flatten(), radii_init])

        # Bounds for centers [0, 1] and radii [0, 0.5]
        bounds = []
        for _ in range(26):
            bounds.extend([(0, 1), (0, 1)]) # x, y
            bounds.append((0, 0.5))          # radius

        # Optimize
        result = minimize(get_loss, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 1000})

        if result.success and -result.fun > best_score:
            best_score = -result.fun
            best_solution = result.x

    if best_solution is None:
        # Fallback to grid if optimization failed
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.1)
        grid_pts = [0.1, 0.3, 0.5, 0.7, 0.9]
        idx = 0
        for y in grid_pts:
            for x in grid_pts:
                if idx < 25:
                    centers[idx] = [x, y]
                    idx += 1
        centers[25] = [0.2, 0.2]
        radii[25] = 0.01 # tiny circle
        return centers, radii, np.sum(radii)

    # Extract best solution
    centers_opt = best_solution[:52].reshape(26, 2)
    radii_opt = best_solution[52:]
    
    # Final validation and cleaning (project to feasible set just in case)
    # If any radius is negative, clamp to 0
    radii_opt = np.maximum(radii_opt, 0)
    
    # Ensure centers are valid
    for i in range(26):
        r = radii_opt[i]
        centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1 - r)
        centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1 - r)

    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii

if __name__ == "__main__":
    import numpy as np
    # Simple test
    centers, radii, total_r = run_packing()
    
    # Validation logic embedded in prompt logic
    n = centers.shape[0]
    valid = True
    for i in range(n):
        if np.isnan(centers[i]).any() or np.isnan(radii[i]):
            valid = False
        if radii[i] < 0:
            valid = False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            valid = False
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                valid = False
    
    print(f"Valid: {valid}")
    print(f"Sum of radii: {total_r}")
