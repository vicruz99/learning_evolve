# sol_000315 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a46c309d) state=84e60be8 sum of radii=2.520608 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def compute_cost_and_penalty(centers, radii, beta):
        """
        Computes the objective value and penalty term.
        Objective: -sum(radii) + beta * penalty
        """
        # Boundary penalties: max(0, violation)^2
        x, y = centers[:, 0], centers[:, 1]
        penalty = 0.0
        
        # Left/Right boundaries
        penalty += np.sum(np.maximum(0, radii - x) ** 2)
        penalty += np.sum(np.maximum(0, x + radii - 1) ** 2)
        
        # Top/Bottom boundaries
        penalty += np.sum(np.maximum(0, radii - y) ** 2)
        penalty += np.sum(np.maximum(0, y + radii - 1) ** 2)
        
        # Pairwise overlap penalties: max(0, r1 + r2 - dist)^2
        # Vectorized pairwise distance
        dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        # Ensure diagonal is 0
        np.fill_diagonal(dist_matrix, 0)
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dist_matrix
        
        # Only consider upper triangle to avoid double counting
        overlap = np.triu(overlap, k=1)
        violations = np.maximum(0, overlap)
        penalty += np.sum(violations ** 2)
        
        objective = -np.sum(radii) + beta * penalty
        return objective, penalty

    def get_initial_configurations():
        configs = []
        
        # 1. Hexagonal-like Grid
        centers_hex = []
        # Try to place points in a dense pattern
        # Radius ~0.1, spacing ~0.2
        # 6 rows, shifting every other
        for i in range(6):
            y = 0.1 + i * (0.1732) # sqrt(3)/2 * 0.2
            x_start = 0.1 if i % 2 == 0 else 0.2
            # How many fit in x?
            # width approx 0.8
            x = x_start
            while x <= 0.9 and len(centers_hex) < n:
                centers_hex.append([x, y])
                x += 0.2
            if len(centers_hex) >= n: break
        
        while len(centers_hex) < n:
            centers_hex.append([np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1])
        centers_hex = np.array(centers_hex[:n])
        configs.append((centers_hex, np.full(n, 0.05)))

        # 2. Square Grid Subset
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.1 + i*0.15, 0.1 + j*0.2])
        pts = np.array(pts)
        indices = np.random.choice(len(pts), n, replace=False)
        configs.append((pts[indices], np.full(n, 0.05)))

        # 3. Random
        centers_rand = np.random.rand(n, 2) * 0.6 + 0.2
        configs.append((centers_rand, np.full(n, 0.05)))
        
        return configs

    best_centers = None
    best_radii = None
    best_sum = -np.inf
    valid_best = None

    # Try multiple optimization runs
    for idx, (init_centers, init_radii) in enumerate(get_initial_configurations()):
        # Vectorize variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)]) # x, y, r

        # Progressive penalty optimization
        current_x = x0.copy()
        betas = [100, 1000, 5000]
        
        for beta in betas:
            def obj(vars):
                c = np.resize(vars[:2*n], (n, 2))
                r = vars[2*n:]
                cost, _ = compute_cost_and_penalty(c, r, beta)
                return cost
            
            # Bounds for minimize
            # L-BFGS-B supports bounds
            res = minimize(obj, current_x, method='L-BFGS-B', bounds=bounds, 
                           options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 2000})
            current_x = res.x
        
        # Extract result
        res_centers = np.resize(current_x[:2*n], (n, 2))
        res_radii = current_x[2*n:]
        
        # Validate and potentially shrink slightly to ensure strict validity
        # Check overlaps and boundary violations
        x, y = res_centers[:, 0], res_centers[:, 1]
        r = res_radii
        
        # Fix boundaries
        r = np.minimum(r, x)
        r = np.minimum(r, 1 - x)
        r = np.minimum(r, y)
        r = np.minimum(r, 1 - y)
        
        # Fix overlaps: if dist < r_i + r_j, scale down radii involved?
        # Simple approach: if overlap, reduce radius of the larger one or both.
        # However, a global reduction is safer for validity.
        # Let's check max overlap
        dist_mat = np.sqrt(np.sum((res_centers[:, np.newaxis] - res_centers[np.newaxis, :]) ** 2, axis=2))
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = np.triu(r_sum - dist_mat, k=1)
        max_viol = np.max(overlap)
        
        if max_viol > 1e-7:
            # Reduce all radii slightly to clear overlaps
            # A safe factor is (dist / (r_i + r_j)) for the worst pair
            # But globally, we can just shrink by a factor.
            # Overlap is small, so shrinking by 1% might be enough.
            # Or compute minimal scaling factor.
            # For simplicity in this context, let's just ensure validity by clipping
            # But clipping might reduce sum significantly.
            # Let's try to resolve overlaps locally?
            # Or just accept the penalty method result if penalty is 0.
            # With high beta, penalty should be ~0.
            pass
            
        current_sum = np.sum(r)
        
        # Final validation check logic (simulation of validate_packing)
        is_valid = True
        # Boundary
        for i in range(n):
            if x[i] - r[i] < -1e-12 or x[i] + r[i] > 1 + 1e-12 or \
               y[i] - r[i] < -1e-12 or y[i] + r[i] > 1 + 1e-12:
                is_valid = False; break
        
        # Overlaps
        if is_valid:
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-12:
                        is_valid = False
                        break
                if not is_valid: break
            
        if is_valid:
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = res_centers.copy()
                best_radii = r.copy()
                valid_best = True
        else:
            # If not strictly valid, maybe shrink radii a tiny bit?
            # Try to find a valid subset by shrinking
            shrink_factor = 0.99
            while shrink_factor > 0.5:
                temp_r = r * shrink_factor
                # Check validity
                temp_valid = True
                for i in range(n):
                    if x[i] - temp_r[i] < 0 or x[i] + temp_r[i] > 1 or \
                       y[i] - temp_r[i] < 0 or y[i] + temp_r[i] > 1:
                        temp_valid = False; break
                if temp_valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < temp_r[i] + temp_r[j] - 1e-12:
                                temp_valid = False; break
                        if not temp_valid: break
                
                if temp_valid:
                    if np.sum(temp_r) > best_sum:
                        best_sum = np.sum(temp_r)
                        best_centers = res_centers.copy()
                        best_radii = temp_r.copy()
                        valid_best = True
                    break
                shrink_factor -= 0.001

    if valid_best:
        return best_centers, best_radii, best_sum
    else:
        # Fallback: uniform small circles
        centers = np.random.rand(n, 2) * 0.5 + 0.25
        radii = np.full(n, 0.01)
        return centers, radii, np.sum(radii)
