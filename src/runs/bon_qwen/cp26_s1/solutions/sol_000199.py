# sol_000199 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1a220354) state=be130d20 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    def objective(vars):
        """
        Objective function: Maximize sum of radii (minimize negative sum).
        Includes penalties for overlaps and boundary violations.
        """
        centers = vars[:2*n].reshape(n, 2)
        radii = vars[2*n:]
        
        cost = -np.sum(radii)
        
        # Boundary penalties
        # Circles must be within [0, 1]
        # x - r >= 0  =>  r - x <= 0
        # x + r <= 1  =>  x + r - 1 <= 0
        # same for y
        
        penalty_weight = 1000.0
        
        # Lower bound constraints (x >= r, y >= r)
        # Violation if r > x => penalty (r - x)
        lower_violation = np.maximum(0, radii - centers[:, 0]) + np.maximum(0, radii - centers[:, 1])
        cost += penalty_weight * np.sum(lower_violation**2)
        
        # Upper bound constraints (x + r <= 1, y + r <= 1)
        # Violation if x + r > 1 => penalty (x + r - 1)
        upper_violation = np.maximum(0, centers[:, 0] + radii - 1.0) + np.maximum(0, centers[:, 1] + radii - 1.0)
        cost += penalty_weight * np.sum(upper_violation**2)
        
        # Overlap penalties
        # dist >= r_i + r_j
        # Violation if dist < r_i + r_j => penalty (r_i + r_j - dist)
        
        # Vectorized distance calculation
        # diffs shape (n, n, 2)
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # radii sum matrix shape (n, n)
        rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount
        overlap = rad_sums - dists
        
        # We only care about i < j, and positive overlap
        # Create mask for upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        positive_overlap = np.maximum(0, overlap)
        total_overlap = np.sum(positive_overlap[mask])
        
        cost += penalty_weight * total_overlap**2
        
        return cost

    def generate_initial_guess():
        """
        Generates a reasonable initial guess based on a hexagonal-like packing.
        """
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Approximate radius for 26 circles
        # Equal packing r approx 0.092. We start slightly smaller to be safe, 
        # but we want to maximize sum, so maybe start around 0.095?
        # However, to ensure valid start, let's use 0.08.
        r_guess = 0.08
        
        # Try to arrange in a hexagonal grid
        # Rows offset
        row_count = 0
        idx = 0
        
        # Estimate rows needed
        # height approx 1.0
        # vertical spacing sqrt(3)/2 * 2r = sqrt(3)*r approx 1.732 * 0.08 = 0.138
        # 1 / 0.138 approx 7 rows.
        
        y = r_guess
        row_idx = 0
        
        while idx < n and y + r_guess <= 1.0:
            # Determine x start for this row
            # Even rows (0, 2, ...) start at r_guess
            # Odd rows (1, 3, ...) start at 2*r_guess (shifted)
            if row_idx % 2 == 0:
                x = r_guess
            else:
                x = 2 * r_guess
            
            while idx < n and x + r_guess <= 1.0:
                centers[idx, 0] = x
                centers[idx, 1] = y
                radii[idx] = r_guess
                idx += 1
                x += 2 * r_guess # Horizontal spacing 2r
            
            y += np.sqrt(3) * r_guess # Vertical spacing sqrt(3)r
            row_idx += 1
            
        # If we didn't fit all, adjust (shouldn't happen with small r)
        if idx < n:
            # Fallback to grid
            for i in range(n):
                r = 0.05
                centers[i, 0] = r + (i % 6) * 2 * r
                centers[i, 1] = r + (i // 6) * 2 * r
                radii[i] = r

        return centers, radii

    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run optimization multiple times with slight perturbations
    for trial in range(5):
        centers_init, radii_init = generate_initial_guess()
        
        # Add small random noise to escape local minima
        centers_init += np.random.uniform(-0.01, 0.01, size=centers_init.shape)
        radii_init += np.random.uniform(-0.005, 0.005, size=radii_init.shape)
        
        # Clip to valid bounds initially
        centers_init = np.clip(centers_init, 0.0, 1.0)
        radii_init = np.clip(radii_init, 0.01, 0.5)
        
        # Flatten variables
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
        for i in range(n):
            bounds.append((0.0, 0.5)) # r
            
        # Optimize
        # L-BFGS-B is good for bound constraints
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < 0: # Should be negative sum
                # Extract solution
                sol_centers = res.x[:2*n].reshape(n, 2)
                sol_radii = res.x[2*n:]
                
                # Verify validity manually to be sure
                # (A quick check)
                valid = True
                
                # Check bounds
                for i in range(n):
                    if sol_radii[i] < 0 or sol_centers[i,0] - sol_radii[i] < -1e-6 or sol_centers[i,0] + sol_radii[i] > 1 + 1e-6:
                        valid = False
                        break
                    if sol_centers[i,1] - sol_radii[i] < -1e-6 or sol_centers[i,1] + sol_radii[i] > 1 + 1e-6:
                        valid = False
                        break
                
                if valid:
                    sum_r = np.sum(sol_radii)
                    if sum_r > best_sum:
                        best_sum = sum_r
                        best_centers = sol_centers
                        best_radii = sol_radii
        except Exception as e:
            print(f"Trial {trial} failed: {e}")
            continue

    if best_centers is None:
        # Fallback
        centers_init, radii_init = generate_initial_guess()
        best_centers = centers_init
        best_radii = radii_init
        best_sum = np.sum(radii_init)

    return best_centers, best_radii, best_sum

# Example usage check
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # validate_packing(centers, radii) would be called by the system
