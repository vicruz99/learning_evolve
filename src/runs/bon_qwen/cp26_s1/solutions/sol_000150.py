# sol_000150 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e234a3e4) state=7128310d sum of radii=1.147802 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_hexagonal_packing(n):
    """
    Generates an initial hexagonal packing for n circles.
    Tries to form rows of roughly equal size.
    """
    # Determine row configuration for n=26
    # A compact shape is best. 5 rows: 5, 6, 5, 6, 4 sums to 26.
    # Or 6, 5, 6, 5, 4?
    # Let's try to pack them tightly.
    rows = [6, 5, 6, 5, 4] # Sum = 26
    # Let's adjust to center them.
    
    # We will place circles in a hexagonal grid.
    # Row spacing in y: sqrt(3)/2 * diameter.
    # But we don't know diameter yet. Let's assume r=0.1 for placement.
    r_est = 0.1
    d = 2 * r_est
    h = d * np.sqrt(3) / 2
    
    centers = []
    current_y = r_est
    for row_idx, count in enumerate(rows):
        # For odd rows (0-indexed), shift x by d/2 if we want staggered
        # Standard hex: row 0 at x=r, row 1 at x=r+d/2 (shifted)
        # But to fit in square, we might want to align left or center.
        # Let's align left for simplicity, then optimize will move them.
        # Actually, centering might be better to avoid boundary issues initially.
        
        # Calculate width of this row
        row_width = count * d 
        # Start x to center the row in [0, 1]
        start_x = (1.0 - row_width) / 2.0 + r_est
        
        # Shift for hex packing
        shift = 0.0
        if row_idx % 2 == 1:
            shift = d / 2.0
            start_x += shift # Shift the whole row? 
            # Actually, standard hex shifts centers by d/2 relative to prev row.
            # If row 0 is centered, row 1 should be centered?
            # No, hex packing shifts the grid.
            # Let's just place them with x increment d.
            # And add shift to start_x.
        
        for i in range(count):
            x = start_x + i * d
            centers.append([x, current_y])
        
        current_y += h
        
    return np.array(centers)

def run_packing():
    n = 26
    
    # 1. Initialization
    # Generate a hexagonal layout
    centers_init = get_hexagonal_packing(n)
    
    # Estimate initial radii. 
    # A grid of 26 circles might fit with r ~ 0.1.
    # But we need to ensure no overlap and inside bounds.
    # Let's scale down to ensure valid start.
    r_init = 0.08 
    
    # Refine initial radii based on distances
    # Simple pass: r_i = min(dist to boundary, min(dist to neighbor)/2)
    # But let's just use a safe small radius.
    radii_init = np.full(n, r_init)
    
    # Check if initial config is valid? 
    # With r=0.08, d=0.16.
    # 5 rows. Height ~ 0.16 + 4*0.16*0.866 ~ 0.16 + 0.55 = 0.71 < 1.
    # Width of 6 circles: 6*0.16 = 0.96 < 1.
    # So r=0.08 is safe.
    
    # 2. Optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Shape (78,)
    
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Constraints
    constraints = []
    
    # Boundary constraints: r <= x, x+r <= 1 => x >= r, x <= 1-r
    # Similarly for y
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
        })

    # Non-overlap constraints: dist >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # This is non-convex.
    # We can use the squared version? No, sqrt is monotonic.
    # But SLSQP handles non-linear constraints.
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            # dist - (ri + rj) >= 0
            def make_constraint(i_idx, j_idx, ii, jj):
                def constraint(v):
                    dx = v[i_idx] - v[j_idx]
                    dy = v[i_idx+1] - v[j_idx+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - (v[i_idx+2] + v[j_idx+2])
                return constraint

            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(idx_xi, idx_xj, idx_ri, idx_rj)
            })

    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    def objective(v):
        sum_r = 0.0
        for i in range(n):
            sum_r += v[3 * i + 2]
        return -sum_r

    # Run optimization
    # SLSQP is suitable for constrained optimization
    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            centers_opt = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
            radii_opt = np.array([res.x[3*i+2] for i in range(n)])
        else:
            # Fallback to initial if optimization fails
            centers_opt = centers_init
            radii_opt = radii_init
            
    except Exception as e:
        # Fallback
        centers_opt = centers_init
        radii_opt = radii_init

    # Post-processing to ensure validity (numerical errors)
    # The solver might produce tiny violations.
    # We can clamp radii to be safe.
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    # r <= dist/2
    
    # A safer way: re-calculate max possible radii for the found centers
    # This ensures validity.
    # However, this might reduce the sum significantly if centers are bad.
    # But centers should be good.
    # Let's compute valid radii for the optimized centers.
    
    valid_radii = np.zeros(n)
    for i in range(n):
        x, y = centers_opt[i]
        # Boundary limits
        r_bound = min(x, 1-x, y, 1-y)
        # Neighbor limits
        r_neigh = np.inf
        for j in range(n):
            if i != j:
                dist = np.sqrt((centers_opt[i,0]-centers_opt[j,0])**2 + (centers_opt[i,1]-centers_opt[j,1])**2)
                r_neigh = min(r_neigh, dist / 2.0)
        valid_radii[i] = min(r_bound, r_neigh)
        
    # Check if valid_radii sum is better?
    # The optimizer should have respected constraints.
    # But due to numerical tolerance, maybe slight violation.
    # Using valid_radii guarantees validity.
    
    # However, if we change radii, we might violate non-overlap with the NEW radii?
    # No, if r_i <= dist/2 for all j, then r_i + r_j <= dist?
    # r_i <= d/2 and r_j <= d/2 implies r_i + r_j <= d.
    # So yes, recomputing radii this way guarantees non-overlap.
    
    # But wait, if we reduce r_i, it might allow r_j to be larger?
    # No, r_j is limited by dist(i,j)/2. If r_i decreases, the limit for r_j (dist/2) doesn't change.
    # But r_j might have been limited by r_i + r_j <= dist.
    # If we enforce r_i <= dist/2 and r_j <= dist/2, we satisfy r_i + r_j <= dist.
    # This is a valid configuration.
    
    # Is it possible that valid_radii sum is much smaller?
    # Only if optimizer failed.
    
    # Let's use the valid radii to be safe.
    radii_final = valid_radii
    
    sum_radii = np.sum(radii_final)
    
    return centers_opt, radii_final, float(sum_radii)
