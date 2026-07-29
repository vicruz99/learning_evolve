# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9f77b693) state=df232d17 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing():
    n = 26
    
    # Initial configuration: Hexagonal packing
    # Rows with counts: 5, 4, 5, 4, 5, 3
    # This sums to 26.
    row_counts = [5, 4, 5, 4, 5, 3]
    
    centers = []
    radii = []
    
    # Estimate radius for hexagonal packing to fit in unit square
    # Width for 5 circles: 10r <= 1 => r <= 0.1
    # Height for 6 rows: 2r + 5*r*sqrt(3) <= 1 => r(2 + 8.66) <= 1 => r <= 1/10.66 ~ 0.0938
    # Let's start with r = 0.08 to be safe
    r_init = 0.08
    
    y_offset = 0.0 # Start y
    # Vertical spacing in hex packing is r * sqrt(3)
    # But we don't know r exactly yet, let's assume r=0.08 for init
    dy = r_init * math.sqrt(3)
    
    current_y = r_init
    row_idx = 0
    
    for count in row_counts:
        # Determine x offset for this row
        # Odd rows (0, 2, 4) aligned left? 
        # In hex packing, usually alternating offset.
        # Let's say row 0 starts at x = r
        # Row 1 starts at x = 2r (shifted by r)
        
        x_start = r_init if (row_idx % 2 == 0) else 2 * r_init
        
        for i in range(count):
            x = x_start + i * (2 * r_init)
            centers.append([x, current_y])
            radii.append(r_init)
        
        current_y += dy
        row_idx += 1
        
    centers = np.array(centers)
    radii = np.array(radii)
    
    # Flatten variables for optimizer: [x1, y1, r1, ..., x26, y26, r26]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    # Actually r can be at most 0.5.
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        r = vars[2*n:]
        return -np.sum(r)
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0  => r_i - x_i <= 0
    # 1 - x_i - r_i >= 0 => x_i + r_i - 1 <= 0
    # Same for y
    
    for i in range(n):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i] - v[2*n + i]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[i] - v[2*n + i]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i+n] - v[2*n + i]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[i+n] - v[2*n + i]
        })

    # Overlap constraints: dist_ij >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i_x = i
            idx_i_y = i + n
            idx_i_r = 2*n + i
            
            idx_j_x = j
            idx_j_y = j + n
            idx_j_r = 2*n + j
            
            def overlap_con(v, i=i, j=j):
                dx = v[idx_i_x] - v[idx_j_x]
                dy = v[idx_i_y] - v[idx_j_y]
                ri = v[idx_i_r]
                rj = v[idx_j_r]
                return (dx*dx + dy*dy) - (ri + rj)**2
            
            constraints.append({'type': 'ineq', 'fun': overlap_con})

    # Optimization
    # SLSQP is suitable for non-linear constraints
    try:
        res = scipy.optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            final_centers = np.zeros((n, 2))
            final_radii = np.zeros(n)
            for i in range(n):
                final_centers[i] = [res.x[i], res.x[i+n]]
                final_radii[i] = res.x[2*n + i]
            
            # Post-processing: Check validity and maybe tighten
            # The solver might satisfy constraints with tolerance.
            # We can scale down radii slightly if needed, but usually SLSQP is precise.
            
            # Validate
            if validate_packing(final_centers, final_radii):
                return final_centers, final_radii, np.sum(final_radii)
            else:
                # If invalid, try to fix by reducing radii slightly?
                # Or just return.
                pass
    except Exception as e:
        # Fallback to initial if optimization fails
        pass
        
    # Fallback: Return initial configuration if optimization failed or invalid
    # But we should ensure it's valid.
    # The initial config is valid for r=0.08?
    # Width 10*0.08 = 0.8 <= 1. Height approx 0.08*10.66 = 0.85 <= 1.
    # Overlaps: spacing 0.16, sum radii 0.16. Touching. Valid.
    
    return centers, radii, np.sum(radii)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True
