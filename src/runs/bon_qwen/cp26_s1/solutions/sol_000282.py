# sol_000282 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8527a4ba) state=c63dc63e sum of radii=2.629754 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Run multiple trials to find the global optimum
    num_trials = 10
    
    for trial in range(num_trials):
        # 1. Generate Initial Configuration
        # Create a hexagonal grid base and add random noise
        centers = np.zeros((n, 2))
        
        # Layout: 5 rows with counts 5, 5, 5, 5, 6
        # This mimics a hexagonal packing structure
        counts = [5, 5, 5, 5, 6]
        y_spacing = 1.0 / 6.0
        y_offset = 0.1
        
        idx = 0
        for row_idx, count in enumerate(counts):
            y_pos = y_offset + row_idx * y_spacing
            if count == 0:
                continue
            
            # Generate x coordinates evenly spaced
            if count == 1:
                x_coords = [0.5]
            else:
                # Distribute points within [0.1, 0.9]
                x_coords = np.linspace(0.1, 0.9, count)
            
            for x in x_coords:
                centers[idx, 0] = x
                centers[idx, 1] = y_pos
                idx += 1
        
        # Add random perturbation to escape exact grid symmetries
        # We keep it small to stay near the dense packing
        centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
        
        # Initial radii (must be small to ensure validity initially)
        radii = np.ones(n) * 0.05

        # 2. Setup Optimization
        # Variables: x1, y1, ..., x26, y26, r1, ..., r26
        # Total 78 variables
        x0 = np.concatenate([centers.flatten(), radii])

        def objective(vars_vec):
            # Maximize sum of radii -> Minimize negative sum
            r = vars_vec[52:]
            return -np.sum(r)

        def constraint_pairwise(vars_vec):
            # Returns d_ij - (r_i + r_j) >= 0
            c = vars_vec[:52].reshape(26, 2)
            r = vars_vec[52:]
            
            # Vectorized distance calculation
            # Shape (26, 26)
            dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=2))
            
            # Extract upper triangle for pairs
            upper_tri_indices = np.triu_indices(n, k=1)
            dists = dist_matrix[upper_tri_indices]
            sum_radii = r[upper_tri_indices[0]] + r[upper_tri_indices[1]]
            
            return dists - sum_radii

        def constraint_boundary_x(vars_vec):
            # Returns (x - r) >= 0 and (1 - x - r) >= 0
            c = vars_vec[:52].reshape(26, 2)
            r = vars_vec[52:]
            return np.concatenate([c[:, 0] - r, (1.0 - c[:, 0]) - r])

        def constraint_boundary_y(vars_vec):
            # Returns (y - r) >= 0 and (1 - y - r) >= 0
            c = vars_vec[:52].reshape(26, 2)
            r = vars_vec[52:]
            return np.concatenate([c[:, 1] - r, (1.0 - c[:, 1]) - r])

        # Bounds for variables
        # x, y in [0, 1], r >= 0
        bounds = []
        for i in range(52):
            bounds.append((0.0, 1.0))
        for i in range(26):
            bounds.append((0.0, None)) # radii non-negative

        constraints = [
            {'type': 'ineq', 'fun': constraint_pairwise},
            {'type': 'ineq', 'fun': constraint_boundary_x},
            {'type': 'ineq', 'fun': constraint_boundary_y},
        ]

        # 3. Run Optimizer
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
            )
            
            if result.success:
                final_vars = result.x
                curr_centers = final_vars[:52].reshape(26, 2)
                curr_radii = final_vars[52:]
                curr_sum = np.sum(curr_radii)
                
                if curr_sum > best_sum:
                    best_centers = curr_centers.copy()
                    best_radii = curr_radii.copy()
                    best_sum = curr_sum
        except Exception:
            continue

    # 4. Post-processing to ensure strict validity
    if best_centers is not None:
        # Clamp centers strictly to [0, 1]
        best_centers = np.clip(best_centers, 1e-12, 1 - 1e-12)
        
        # Correct any minor overlaps due to numerical tolerance
        # We reduce radii slightly if needed
        for i in range(n):
            # Boundary correction
            r = best_radii[i]
            x, y = best_centers[i]
            max_r_bound = min(x, 1-x, y, 1-y)
            if r > max_r_bound:
                best_radii[i] = max_r_bound - 1e-12

        # Pairwise correction (very conservative shrink to ensure 1e-12 gap)
        # This is expensive but necessary for strict validation
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                sum_r = best_radii[i] + best_radii[j]
                if sum_r > dist:
                    # Reduce radii proportionally to fit
                    overlap = sum_r - dist
                    # Split the reduction
                    reduction = overlap / 2.0 + 1e-12
                    best_radii[i] -= reduction / 2.0
                    best_radii[j] -= reduction / 2.0

        # Ensure no negative radii after corrections
        best_radii = np.maximum(best_radii, 0.0)

    return best_centers, best_radii, np.sum(best_radii)
