# sol_000271 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=59e1710b sum of radii=2.457695 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    # Initial guess: Hexagonal grid approximation
    # We try to fit 26 circles. 5x5 is 25. 
    # A hexagonal packing of 5 rows might look like 6, 5, 6, 5, 4 (sum 26) or similar.
    # Let's try a randomized grid initialization first to break symmetry, 
    # or a structured one. Structured is usually better for local opt.
    
    # Layout: 5 rows. 
    # Row 0: 6 circles
    # Row 1: 5 circles
    # Row 2: 6 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Total: 26.
    
    # Initial radius guess. 
    # Width constraint for 6 circles: 12r <= 1 => r <= 0.0833
    # Width constraint for 4 circles: 8r <= 1 => r <= 0.125
    # Let's start with r = 0.08 to be safe.
    r_init = 0.08
    
    centers = []
    radii = []
    
    row_counts = [6, 5, 6, 5, 4]
    
    current_y = r_init
    for i, count in enumerate(row_counts):
        # Stagger rows
        offset = 0.0
        if i % 2 == 1:
            offset = r_init # Shift by r (half diameter) for hex packing
            
        # Spacing
        # Available width 1. 
        # We want to center the row.
        # Total width of circles = count * 2 * r_init
        # But in hex packing, horizontal spacing is 2r.
        # If we just place them with spacing 2r, width is (count-1)*2r + 2r = 2*count*r.
        # We can scale spacing to fit width 1.
        
        # Let's just distribute them uniformly in [r_init, 1-r_init]
        # But for hex packing, x-coords should be related.
        
        # Simple approach: distribute centers in [r_init, 1-r_init]
        # If row is shifted, start at r_init + offset?
        # If offset > 0, first center at r_init + offset.
        # Last center must be <= 1 - r_init.
        
        start_x = r_init + offset
        end_x = 1.0 - r_init
        
        if start_x > end_x:
            # Fallback if offset makes it impossible
            start_x = r_init
            end_x = 1.0 - r_init
            offset = 0.0 # Not really offsetting, just warning
            
        # Generate x coords
        if count == 1:
            x_coords = [0.5]
        else:
            # Linear spacing
            x_coords = np.linspace(start_x, end_x, count)
            
        for x in x_coords:
            centers.append([x, current_y])
            radii.append(r_init)
            
        # Move to next row
        # Vertical spacing for hex packing is sqrt(3)*r
        current_y += np.sqrt(3) * r_init

    centers = np.array(centers)
    radii = np.array(radii)
    
    # Now optimize
    # We will optimize centers and radii.
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Total 78 variables.
    
    # Define objective and constraints
    def objective(vars):
        # vars: [x1, y1, r1, ..., xn, yn, rn]
        r_vals = vars[2::3]
        return -np.sum(r_vals) # Minimize negative sum

    def constraints(vars):
        c_list = []
        c_list.append(opt.NonlinConstraint(lambda v: v[2::3], lb=1e-6, ub=1.0)) # Radii bounds
        
        # Boundary constraints: x >= r, x <= 1-r => x - r >= 0, 1 - x - r >= 0
        # y >= r, y <= 1-r => y - r >= 0, 1 - y - r >= 0
        # Linear constraints in terms of vars? 
        # x_i - r_i >= 0 -> v[3i] - v[3i+2] >= 0
        # 1 - x_i - r_i >= 0 -> 1 - v[3i] - v[3i+2] >= 0
        # Same for y.
        
        # These are linear. We can add them as linear constraints or bounds?
        # Bounds are easier. But r is variable.
        # Let's just use non-linear constraints for simplicity in scipy.
        
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # x >= r
        c_list.append(opt.NonlinConstraint(lambda v: v[0::3] - v[2::3], lb=0))
        # x <= 1 - r  => x + r <= 1
        c_list.append(opt.NonlinConstraint(lambda v: 1.0 - (v[0::3] + v[2::3]), lb=0))
        # y >= r
        c_list.append(opt.NonlinConstraint(lambda v: v[1::3] - v[2::3], lb=0))
        # y <= 1 - r
        c_list.append(opt.NonlinConstraint(lambda v: 1.0 - (v[1::3] + v[2::3]), lb=0))
        
        return c_list

    # Overlap constraints are non-linear and numerous.
    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # This is hard for scipy's SLSQP if we add all 26*25/2 = 325 constraints.
    # Maybe use a penalty method instead?
    
    # Penalty Method
    # Minimize -Sum(r) + Penalty(overlaps) + Penalty(boundary)
    
    def loss_function(vars):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        obj = -np.sum(r)
        
        penalty = 0.0
        K = 1000.0 # Penalty strength
        
        # Boundary penalties
        # If x < r, penalty. If x > 1-r, penalty.
        # Soft boundary: max(0, r - x)^2 + max(0, x + r - 1)^2
        penalty += K * np.sum(np.maximum(0, r - x)**2)
        penalty += K * np.sum(np.maximum(0, x + r - 1)**2)
        penalty += K * np.sum(np.maximum(0, r - y)**2)
        penalty += K * np.sum(np.maximum(0, y + r - 1)**2)
        
        # Overlap penalties
        # For all pairs i < j: max(0, ri + rj - dist)^2
        # Vectorized computation
        n = len(r)
        # Create matrices
        X = x.reshape(-1, 1) - x.reshape(1, -1)
        Y = y.reshape(-1, 1) - y.reshape(1, -1)
        R = r.reshape(-1, 1) + r.reshape(1, -1)
        
        dist = np.sqrt(X**2 + Y**2)
        # Upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlaps = np.maximum(0, R[mask] - dist[mask])
        penalty += K * np.sum(overlaps**2)
        
        return obj + penalty

    # Initial vars
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = centers[:, 0]
    vars_init[1::3] = centers[:, 1]
    vars_init[2::3] = radii
    
    # Optimize
    # Use Nelder-Mead or L-BFGS-B. L-BFGS-B handles bounds but not complex constraints well.
    # Penalty function turns it into unconstrained (or bounded).
    # Let's use L-BFGS-B with bounds on vars?
    # Bounds: x in [0, 1], y in [0, 1], r in [0, 0.5]
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    result = opt.minimize(loss_function, vars_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    
    best_vars = result.x
    best_centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    best_radii = best_vars[2::3]
    
    # Post-processing: ensure validity by shrinking radii if overlaps exist
    # This handles numerical errors or imperfect convergence
    max_iter = 50
    for _ in range(max_iter):
        overlap_found = False
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            
            # Boundary check
            min_dist_boundary = min(x, 1-x, y, 1-y)
            if r > min_dist_boundary:
                best_radii[i] = min_dist_boundary
                overlap_found = True
            
            # Overlap check with others
            for j in range(i + 1, n):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                sum_r = best_radii[i] + best_radii[j]
                if dist < sum_r - 1e-9:
                    # Overlap, reduce radii
                    # Simple heuristic: reduce both equally or proportional
                    # Just ensure sum_r <= dist
                    excess = sum_r - dist
                    if excess > 0:
                        reduction = excess / 2
                        best_radii[i] -= reduction
                        best_radii[j] -= reduction
                        # Ensure non-negative
                        if best_radii[i] < 0: best_radii[i] = 0
                        if best_radii[j] < 0: best_radii[j] = 0
                        overlap_found = True
        
        if not overlap_found:
            break
            
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii
