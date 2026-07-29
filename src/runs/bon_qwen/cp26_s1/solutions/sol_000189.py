# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4dd6d242) state=d94c176a sum of radii=2.539999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n_circles = 26
    best_centers = None
    best_radii = None
    max_sum_radii = 0.0

    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        centers = params[0::3].reshape(-1, 2)
        radii = params[1::3]
        return -np.sum(radii)

    def inequality_constraints(params):
        centers = params[0::3].reshape(-1, 2)
        radii = params[1::3]
        cons = []
        
        # Boundary constraints
        for i in range(n_circles):
            cons.append(centers[i, 0] - radii[i])
            cons.append(centers[i, 1] - radii[i])
            cons.append(1 - centers[i, 0] - radii[i])
            cons.append(1 - centers[i, 1] - radii[i])
            
        # Overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                cons.append(dist - radii[i] - radii[j])
                
        return np.array(cons)

    def generate_initial_guess(seed_val):
        np.random.seed(seed_val)
        # Hexagonal packing initialization
        initial_positions = []
        r_est = 0.1 
        row = 0
        for i in range(n_circles):
            if i % 2 == 0:
                x = r_est + (i // 2) * 2 * r_est
            else:
                x = r_est + ((i - 1) // 2) * 2 * r_est + r_est
            
            # Adjust row based on i to form a shape
            # Trying to fit in a square-ish block
            cols_in_row = 5
            y = r_est + (i // cols_in_row) * r_est * np.sqrt(3)
            initial_positions.append([x, y])
            
        # Add random noise
        noise = np.random.uniform(-0.05, 0.05, size=(n_circles, 2))
        initial_positions = np.array(initial_positions) + noise
        
        # Clip to valid range for centers (keep some margin)
        initial_positions = np.clip(initial_positions, 0.1, 0.9)
        
        # Flatten and add radii
        params = np.zeros(n_circles * 3)
        params[0::3] = initial_positions[:, 0]
        params[1::3] = initial_positions[:, 1]
        params[2::3] = r_est 
        return params

    # Try multiple random seeds to find a good configuration
    seeds = [42, 123, 456, 789, 1024, 2024, 0, 999, 7, 11]
    
    for seed in seeds:
        try:
            initial_params = generate_initial_guess(seed)
            
            # Constraints for SLSQP
            bounds = [(0.0, 1.0)] * (2 * n_circles) + [(0.0, 0.5)] * n_circles
            cons = {'type': 'ineq', 'fun': inequality_constraints}
            
            # Use L-BFGS-B for bounds and SLSQP for constraints if needed, 
            # but SLSQP handles both.
            result = minimize(
                objective, 
                initial_params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                final_params = result.x
                centers = final_params[0::3].reshape(-1, 2)
                radii = final_params[1::3]
                
                # Ensure non-negative radii
                radii = np.maximum(radii, 1e-9)
                
                # Recalculate sum
                current_sum = np.sum(radii)
                
                # Verify validity before updating best
                # Quick check for overlaps
                valid = True
                for i in range(n_circles):
                    if radii[i] <= 0: valid = False; break
                    if centers[i, 0] < radii[i] or centers[i, 0] > 1 - radii[i]: valid = False; break
                    if centers[i, 1] < radii[i] or centers[i, 1] > 1 - radii[i]: valid = False; break
                    for j in range(i + 1, n_circles):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < radii[i] + radii[j] - 1e-10:
                            valid = False
                            break
                    if not valid: break
                
                if valid and current_sum > max_sum_radii:
                    max_sum_radii = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
        except Exception:
            continue

    # Fallback to a valid grid packing if optimization fails or returns poor result
    if best_centers is None or max_sum_radii < 2.5:
        grid_centers = []
        grid_radii = []
        r = 0.1
        for i in range(5):
            for j in range(5):
                grid_centers.append([r + i * 2 * r, r + j * 2 * r])
                grid_radii.append(r)
        # Add a tiny circle for the 26th in a gap
        grid_centers.append([0.2, 0.2]) # Center of a gap
        grid_radii.append(0.04) # Approx radius that fits
        
        best_centers = np.array(grid_centers)
        best_radii = np.array(grid_radii)
        max_sum_radii = np.sum(best_radii)

    # Final verification and adjustment
    # Ensure strict non-overlap
    for i in range(n_circles):
        best_radii[i] = min(best_radii[i], best_centers[i, 0], 1 - best_centers[i, 0], 
                            best_centers[i, 1], 1 - best_centers[i, 1])
        
    # Pairwise overlap resolution (simple shrink)
    for _ in range(5): # Few iterations to tighten
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(best_centers[i] - best_centers[j])
                sum_r = best_radii[i] + best_radii[j]
                if dist < sum_r:
                    # Shrink both slightly to resolve
                    overlap = (sum_r - dist) / 2 + 1e-7
                    best_radii[i] = max(1e-9, best_radii[i] - overlap)
                    best_radii[i] = max(1e-9, best_radii[j] - overlap) # Correction

    # Re-verify boundaries after shrinking
    for i in range(n_circles):
        max_r = min(best_centers[i, 0], 1 - best_centers[i, 0], 
                    best_centers[i, 1], 1 - best_centers[i, 1])
        best_radii[i] = min(best_radii[i], max_r)

    max_sum_radii = float(np.sum(best_radii))
    
    return best_centers, best_radii, max_sum_radii
