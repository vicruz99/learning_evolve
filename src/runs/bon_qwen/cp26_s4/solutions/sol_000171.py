# sol_000171 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 403fd447) state=ad2fcfe9 sum of radii=2.143789 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- 1. Initialization: Hexagonal Grid ---
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Estimate a good initial radius for 26 circles
    # A loose hexagonal packing allows r ~ 0.09
    r_init = 0.09
    
    # Create 6 rows. 5 rows of 5 circles = 25. We need 1 more.
    # Let's try to fit 26 in a 6-row hexagonal pattern.
    # Rows: 5, 5, 5, 5, 5, 1? Or distribute better?
    # Actually, 5 rows of 5 is 25. Adding a 26th is hard in strict grid.
    # Let's try a 6x5 grid logic but hexagonal.
    # Rows 0-4: 5 circles each. Row 5: 1 circle?
    # Or maybe 5, 5, 5, 4, 4, 3? 
    # Let's just place them in a dense hexagonal pattern and let the optimizer sort it out.
    
    # We will place circles in rows.
    # Row height = r * sqrt(3)
    # Let's aim for 6 rows.
    
    rows_config = [5, 5, 5, 5, 5, 1] # Total 26
    # To make it more uniform, maybe [5, 5, 5, 5, 4, 2]?
    # Let's try to distribute 26 into 6 rows as evenly as possible: 5, 5, 5, 4, 4, 3?
    # 5+5+5+4+4+3 = 26.
    rows_config = [5, 5, 5, 4, 4, 3]
    
    y_step = 1.0 / 6.0 # Rough spacing
    # Better vertical spacing for hex packing:
    # We have 6 rows. Height 1. 
    # 2r + 5 * r*sqrt(3) <= 1  => r(2 + 5*1.732) <= 1 => r(10.66) <= 1 => r <= 0.093
    # Let's use r=0.09 for initialization.
    
    r_temp = 0.09
    h = r_temp * np.sqrt(3)
    y_start = r_temp
    
    idx = 0
    for i, count in enumerate(rows_config):
        y = y_start + i * h
        # x positions
        # Row i offset by r if i is odd?
        offset = r_temp if i % 2 == 1 else 0
        x_start = r_temp + offset
        
        # Distribute count circles in [x_start, 1 - r_temp]
        width_available = 1 - 2 * r_temp - offset
        # If count is 1, center it?
        if count == 1:
            x = 0.5
            centers[idx] = [x, y]
            radii[idx] = r_temp
            idx += 1
        else:
            # Spacing
            if count > 1:
                spacing = width_available / (count - 1) if width_available > 0 else 0
                # If spacing is too small, just pack them tightly
                if spacing < 2 * r_temp:
                     # Just place them evenly in available space
                    x_coords = np.linspace(r_temp + offset, 1 - r_temp, count)
                else:
                    x_coords = np.linspace(r_temp + offset, 1 - r_temp, count)
                
                for x in x_coords:
                    centers[idx] = [x, y]
                    radii[idx] = r_temp
                    idx += 1
            else:
                 # Should not happen based on config
                 pass

    # --- 2. Force-Directed Layout (Repulsion + Expansion) ---
    # We will iteratively try to increase radii and fix overlaps.
    
    # Increase radii slowly
    current_r = np.ones(n) * 0.05
    centers = centers.copy() # Reset centers to initialization
    radii[:] = current_r
    
    # Random jitter to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Optimization loop
    for step in range(500):
        # Try to increase radii
        # We want to maximize sum, so let's just bump them up
        # But we need to solve for valid configuration.
        # Let's use a simple repulsive force simulation.
        
        forces = np.zeros_like(centers)
        r_grad = np.ones(n) # Gradient for radii (want to increase)
        
        # Check overlaps and boundary
        for i in range(n):
            # Boundary forces
            # Push center away from boundary if r is close
            if centers[i, 0] - radii[i] < 0:
                forces[i, 0] += 1.0 - (centers[i, 0] - radii[i]) * 10
            if centers[i, 0] + radii[i] > 1:
                forces[i, 0] -= 1.0 - (1 - (centers[i, 0] + radii[i])) * 10
            if centers[i, 1] - radii[i] < 0:
                forces[i, 1] += 1.0 - (centers[i, 1] - radii[i]) * 10
            if centers[i, 1] + radii[i] > 1:
                forces[i, 1] -= 1.0 - (1 - (centers[i, 1] + radii[i])) * 10
        
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            r_sum = radii[i] + radii[j]
            
            if dist < r_sum and dist > 1e-9:
                # Overlap!
                # Repulsive force proportional to overlap
                overlap = r_sum - dist
                force_mag = overlap * 5.0 # Strong repulsion
                f_vec = (diff / dist) * force_mag
                forces[i] += f_vec
                forces[j] -= f_vec
            # If dist is very small, add random push
            elif dist < 1e-5:
                forces[i] += np.random.randn(2) * 0.1
                forces[j] -= np.random.randn(2) * 0.1

        # Update centers
        lr_pos = 0.05
        centers += lr_pos * forces
        centers = np.clip(centers, 0.01, 0.99)
        
        # Update radii: try to expand
        # If no overlap, expand. If overlap, shrink?
        # A simple heuristic: 
        # r_i = min(distance to neighbors / 2, distance to boundary)
        # But this is static. We want dynamic growth.
        
        # Let's just try to increase radii by a small amount if valid?
        # Or better: calculate max possible radius for each circle based on current neighbors
        # and move towards it.
        
        new_radii = radii.copy()
        for i in range(n):
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # Constraint: r_i + r_j <= d  => r_i <= d - r_j
                # We don't know optimal r_j, but we can use current r_j as a hint
                # Or just ensure non-overlap with current r_j?
                # If we increase r_i, we might overlap.
                pass
            
            # Simple expansion:
            # Check if we can increase r_i
            # Limit by boundary
            limit_bound = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            # Limit by neighbors (assuming neighbors stay same size for this step)
            limit_neigh = 1.0
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                limit_neigh = min(limit_neigh, d - radii[j])
            
            target_r = min(limit_bound, limit_neigh)
            # Smoothly approach target
            new_radii[i] = radii[i] + 0.1 * (target_r - radii[i])
            
        radii[:] = new_radii
        
        # Decay learning rate? No, keep it steady for packing
        
    # --- 3. Local Optimization with SLSQP ---
    # Flatten variables
    x0 = np.concatenate([centers.flatten(), radii])
    
    def objective(vars):
        # Minimize negative sum of radii
        c = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        return -np.sum(r)

    def constraints(vars):
        c = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        cons = []
        
        # Boundary constraints
        for i in range(n):
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n+i]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i] - v[2*n+i]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n+i]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[2*i+1] - v[2*n+i]})
            
            # Radius non-negative (handled by bounds usually, but added for safety)
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n+i]})

        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                # dist^2 >= (r_i + r_j)^2
                # dist^2 - (r_i + r_j)^2 >= 0
                # vars indices:
                # x_i: 2*i, y_i: 2*i+1
                # x_j: 2*j, y_j: 2*j+1
                # r_i: 2*n+i, r_j: 2*n+j
                
                def overlap_con(v, i=i, j=j):
                    xi, yi = v[2*i], v[2*i+1]
                    xj, yj = v[2*j], v[2*j+1]
                    ri, rj = v[2*n+i], v[2*n+j]
                    dist_sq = (xi-xj)**2 + (yi-yj)**2
                    r_sum_sq = (ri + rj)**2
                    return dist_sq - r_sum_sq
                
                cons.append({'type': 'ineq', 'fun': overlap_con})
        
        return cons

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (max radius 0.5)

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            centers_opt = res.x[:2*n].reshape(n, 2)
            radii_opt = res.x[2*n:]
        else:
            # Fallback to previous if optimization failed
            centers_opt = centers
            radii_opt = radii
    except Exception as e:
        centers_opt = centers
        radii_opt = radii

    # Final validation and cleanup
    # Ensure non-negative radii
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Clip centers slightly to ensure validity (add epsilon)
    # But optimization should have handled it.
    
    # Re-center if necessary? No, just return.
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
