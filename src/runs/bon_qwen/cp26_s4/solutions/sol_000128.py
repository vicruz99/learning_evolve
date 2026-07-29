# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cd61366d) state=4e6b1a52 sum of radii=2.250000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        # Shape: (n_circles * 3,)
        centers = vars[0::3].reshape(n_circles, 2)
        radii = vars[1::3]
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj_val = -np.sum(radii)
        
        # Penalty for radii < 0
        radii_pen = 1000.0 * np.sum(np.minimum(0, radii))
        obj_val += radii_pen
        
        # Penalty for circles outside unit square
        # x - r >= 0 => r - x <= 0
        # x + r <= 1 => x + r - 1 <= 0
        # y - r >= 0 => r - y <= 0
        # y + r <= 1 => y + r - 1 <= 0
        bound_pen = 0.0
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            # We use a smooth penalty: max(0, violation)^2
            v1 = np.maximum(0, r - x)
            v2 = np.maximum(0, x + r - 1)
            v3 = np.maximum(0, r - y)
            v4 = np.maximum(0, y + r - 1)
            bound_pen += 1000.0 * (v1**2 + v2**2 + v3**2 + v4**2)
        obj_val += bound_pen
        
        # Penalty for overlaps
        # dist >= r_i + r_j => dist - r_i - r_j >= 0
        # violation = max(0, r_i + r_j - dist)
        overlap_pen = 0.0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                min_dist = radii[i] + radii[j]
                violation = np.maximum(0, min_dist - dist)
                # Strong penalty for overlap
                overlap_pen += 2000.0 * violation**2
        obj_val += overlap_pen
        
        return obj_val

    def get_initial_guess(seed=0):
        rng = np.random.default_rng(seed)
        # Create a hexagonal-like initial packing
        # Approximate radius for 26 circles
        # 5x5 grid gives r=0.1. 26 circles need slightly smaller r or better packing.
        # Let's start with r=0.09 to ensure feasibility
        r_init = 0.09
        
        centers = np.zeros((n_circles, 2))
        radii = np.full(n_circles, r_init)
        
        idx = 0
        # Hexagonal packing pattern
        # Rows with y spacing sqrt(3)*r
        # Even rows shifted by r? No, shifted by half width? 
        # Standard hex: row i at y = 2r + i * sqrt(3)*r. 
        # But we want to fit in [0,1].
        
        # Let's just place them in a perturbed grid first
        # 6 rows of 4 or 5?
        # 26 circles. 6 rows: 5, 4, 5, 4, 5, 3? 
        # Let's try to fill a grid
        
        rows = 6
        cols = 5
        count = 0
        for r_idx in range(rows):
            y = 0.1 + r_idx * 0.15 # Rough spacing
            if y > 0.9: break
            # Shift x for alternating rows
            x_offset = 0.0 if r_idx % 2 == 0 else 0.05
            for c_idx in range(cols):
                if count >= n_circles: break
                x = 0.1 + x_offset + c_idx * 0.2
                if x < 1.0 and y < 1.0:
                    centers[count] = [x, y]
                    count += 1
        
        # Fill remaining if any
        while count < n_circles:
            centers[count] = [0.5, 0.5]
            count += 1
            
        # Add some randomness to escape symmetry
        centers += rng.normal(0, 0.01, size=centers.shape)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Flatten for optimizer
        vars_init = np.zeros(n_circles * 3)
        vars_init[0::3] = centers[:, 0]
        vars_init[1::3] = centers[:, 1]
        vars_init[2::3] = radii
        
        return vars_init

    # Try multiple seeds to find global optimum
    for seed in range(10):
        x0 = get_initial_guess(seed)
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='BFGS',
                options={'maxiter': 5000, 'ftol': 1e-12}
            )
            
            # Extract results
            centers = res.x[0::3].reshape(n_circles, 2)
            radii = res.x[2::3]
            
            # Check validity (penalty should be 0)
            if res.fun < -np.sum(radii) + 1e-6: # Approximate check
                # Calculate actual sum
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    # Verify no overlaps manually to be safe before updating best
                    valid = True
                    for i in range(n_circles):
                        x, y = centers[i]
                        r = radii[i]
                        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                            valid = False
                            break
                        for j in range(i + 1, n_circles):
                            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                            if dist < radii[i] + radii[j] - 1e-9:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        best_sum = current_sum
                        best_centers = centers.copy()
                        best_radii = radii.copy()
        except Exception:
            continue

    # If optimization failed to find a valid packing (unlikely with penalties), fallback to simple grid
    if best_centers is None:
        centers = np.zeros((n_circles, 2))
        radii = np.full(n_circles, 0.0)
        # Simple 5x5 grid with small radius
        r = 0.09
        idx = 0
        for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
                if idx < n_circles:
                    centers[idx] = [x, y]
                    radii[idx] = r
                    idx += 1
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    return best_centers, best_radii, best_sum
