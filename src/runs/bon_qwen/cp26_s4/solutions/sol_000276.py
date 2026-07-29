# sol_000276 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=92139dc2 sum of radii=2.621548 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Helper to create initial valid packing (Hexagonal grid)
    def get_initial_hexagonal_packing(seed_offset=0):
        # Hexagonal packing parameters
        # We want to fit circles. Let's assume a radius around 0.08-0.09 initially.
        # Spacing s = 2*r.
        # Vertical spacing dy = s * sqrt(3)/2.
        
        # Let's try to pack tightly. 
        # Approximate density suggests r ~ 0.1 for 25 circles. 
        # For 26, maybe slightly less.
        r_init = 0.06 # Start small to be safe, optimizer will grow them
        
        s = 2 * r_init
        dy = s * np.sqrt(3) / 2
        
        centers = []
        
        # Generate hex grid points
        # Row 0: y = r_init, x = r_init, r_init + s, ...
        # Row 1: y = r_init + dy, x = r_init + s/2, ...
        
        y_curr = r_init
        row_idx = 0
        
        while len(centers) < n:
            x_curr = r_init if row_idx % 2 == 0 else r_init + s / 2
            
            while x_curr + r_init <= 1.0 - 1e-9: # Ensure inside boundary
                if len(centers) < n:
                    centers.append([x_curr, y_curr])
                x_curr += s
            
            y_curr += dy
            row_idx += 1
            
            # Safety break if y grows too large
            if y_curr + r_init > 1.0 + 0.1: 
                break
        
        # If we didn't get enough points (should not happen with r=0.06), pad with random
        while len(centers) < n:
            centers.append(np.random.rand(2) * 0.8 + 0.1)
            
        centers = np.array(centers[:n])
        radii = np.ones(n) * r_init
        
        return centers, radii

    def objective(vars):
        # vars: [x1, ..., xn, y1, ..., yn, r1, ..., rn]
        r = vars[2*n:]
        return -np.sum(r)

    def constraints(vars):
        # vars: [x1, ..., xn, y1, ..., yn, r1, ..., rn]
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        
        cons = []
        
        # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        # Flattened:
        # x - r >= 0
        cons.extend(x - r)
        # 1 - x - r >= 0
        cons.extend(1.0 - x - r)
        # y - r >= 0
        cons.extend(y - r)
        # 1 - y - r >= 0
        cons.extend(1.0 - y - r)
        
        # Overlap constraints: dist_ij - (r_i + r_j) >= 0
        # Vectorized distance calculation
        # X_diff shape (n, n), Y_diff shape (n, n)
        # But we only need upper triangle
        
        # To avoid creating huge arrays in every iteration, we can compute manually or use broadcasting
        # Broadcasting is efficient in numpy
        X = x.reshape(-1, 1) - x.reshape(1, -1) # (n, n)
        Y = y.reshape(-1, 1) - y.reshape(1, -1) # (n, n)
        
        dist_matrix = np.sqrt(X**2 + Y**2)
        
        # Radius sum matrix
        R_sum = r.reshape(-1, 1) + r.reshape(1, -1)
        
        # Constraint value
        constraint_val_matrix = dist_matrix - R_sum
        
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        overlap_cons = constraint_val_matrix[triu_indices]
        
        cons.extend(overlap_cons)
        
        return np.array(cons)

    # Bounds for variables
    # x, y in [0, 1] (though effectively constrained by r)
    # r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    best_result = None
    best_sum_r = -1
    
    # Try multiple starts
    # 1. Hexagonal grid
    # 2. Random valid packings
    
    configs_to_try = 5
    
    for i in range(configs_to_try):
        if i == 0:
            centers, radii = get_initial_hexagonal_packing()
        else:
            # Random valid initialization
            # Place points randomly, ensure minimal distance
            centers = np.random.rand(n, 2)
            radii = np.ones(n) * 0.02
            # Simple repulsion to separate
            for _ in range(100):
                dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                # Force apart if too close (simplified)
                # Not strictly necessary if optimizer handles it, but helps
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Options for SLSQP
        options = {
            'maxiter': 500,
            'ftol': 1e-10,
            'disp': False
        }
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints}, 
                           options=options)
            
            if res.success or res.fun < -best_sum_r: # Since we minimize -sum_r, lower fun is better
                current_sum_r = -res.fun
                if current_sum_r > best_sum_r:
                    best_sum_r = current_sum_r
                    best_result = res
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        centers, radii = get_initial_hexagonal_packing()
        return centers, radii, np.sum(radii)

    # Extract best solution
    final_x = best_result.x[:n]
    final_y = best_result.x[n:2*n]
    final_r = best_result.x[2*n:]
    
    final_centers = np.column_stack((final_x, final_y))
    
    # Post-processing to strictly satisfy validation tolerances
    # The solver might have slight inaccuracies. 
    # We can shrink radii slightly to ensure no overlap violation (1e-12).
    # But validation allows 1e-12 tolerance.
    
    # Let's verify and fix any minor violations by shrinking radii if necessary
    # This is a safety measure.
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j]) ** 2))
            req_dist = final_r[i] + final_r[j]
            if dist < req_dist - 1e-12:
                # Violation detected, shrink both radii equally
                overlap = req_dist - dist
                shrink = overlap / 2 + 1e-10
                final_r[i] -= shrink
                final_r[j] -= shrink
                
    # Check boundaries
    for i in range(n):
        x, y = final_centers[i]
        r = final_r[i]
        # x - r >= 0 => r <= x
        if r > x + 1e-12:
            final_r[i] = x
        # x + r <= 1 => r <= 1 - x
        if r > 1 - x + 1e-12:
            final_r[i] = 1 - x
        # y - r >= 0
        if r > y + 1e-12:
            final_r[i] = y
        # y + r <= 1
        if r > 1 - y + 1e-12:
            final_r[i] = 1 - y
            
    # Ensure non-negative radii
    final_r = np.maximum(final_r, 0)
    
    return final_centers, final_r, np.sum(final_r)
