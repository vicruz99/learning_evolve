# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=3f466ec8 sum of radii=2.179181 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # Arrangement of rows: 6, 5, 6, 5, 4 circles to total 26.
    # This mimics a dense hexagonal packing.
    rows_counts = [6, 5, 6, 5, 4]
    centers = []
    
    # Vertical spacing and starting Y
    # We have 5 rows. We can space them evenly.
    y_step = 0.8 / 4.0 # Spread over middle 80% initially
    y_start = 0.1
    
    for row_idx, count in enumerate(rows_counts):
        y = y_start + row_idx * y_step
        
        # Horizontal spacing
        # Stagger rows for hexagonal effect
        # Even rows (0, 2, 4) aligned one way, odd (1, 3) shifted
        # Actually, standard hex packing shifts by half width.
        # Let's just distribute them evenly in x for simplicity, 
        # the optimizer will fix the x-alignment.
        
        # Distribute 'count' circles in [0.1, 0.9]
        x_coords = np.linspace(0.1, 0.9, count)
        
        for x in x_coords:
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Initialize radii small
    radii = np.full(n, 0.04)
    
    # 2. Force-Directed Simulation
    # This phase expands radii and resolves overlaps roughly.
    max_iters = 2000
    lr_pos = 0.05  # Learning rate for positions
    lr_rad = 0.002 # Radius growth rate
    
    for step in range(max_iters):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    # Repulsive force magnitude proportional to overlap
                    f_mag = overlap * 10.0 
                    fx = (dx / dist) * f_mag
                    fy = (dy / dist) * f_mag
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < 1e-8:
                    # Avoid division by zero, push randomly
                    forces[i, 0] += 1.0
                    forces[j, 0] -= 1.0

        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Walls push center away
            if x < r:
                forces[i, 0] += (r - x) * 20.0
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 20.0
            if y < r:
                forces[i, 1] += (r - y) * 20.0
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 20.0
                
        # Update positions
        # Decay learning rate slightly
        current_lr = lr_pos * (1.0 / (1.0 + step * 0.0005))
        centers += forces * current_lr
        
        # Clip to [0, 1] to prevent NaNs
        centers = np.clip(centers, 0, 1)
        
        # Grow radii
        # Only grow if overlaps are minimal or forces are pushing apart
        # Here we just grow slowly; forces will push centers apart to make room.
        radii += lr_rad
        
        # Cap radii at 0.5 (cannot be larger)
        radii = np.minimum(radii, 0.5)

    # 3. Scipy Optimization Refinement
    # Maximize sum of radii with constraints
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Total 3 * n variables.
    
    # Flatten variables for scipy
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    def objective(vars):
        # Maximize sum of radii -> minimize negative sum
        r = vars[2::3]
        return -np.sum(r)

    def constraints(vars):
        cons = []
        centers = vars[0::3], vars[1::3]
        cx = np.array(centers[0])
        cy = np.array(centers[1])
        r = vars[2::3]
        
        # Boundary constraints: r <= x <= 1-r  =>  x-r >= 0, 1-x-r >= 0
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]}) # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y - r >= 0
            
        # Overlap constraints: dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_i_x, idx_i_y, idx_i_r = 3*i, 3*i+1, 3*i+2
                idx_j_x, idx_j_y, idx_j_r = 3*j, 3*j+1, 3*j+2
                
                def overlap_con(v, i=i, j=j):
                    dx = v[idx_i_x] - v[idx_j_x]
                    dy = v[idx_i_y] - v[idx_j_y]
                    di = v[idx_i_r] + v[idx_j_r]
                    return dx*dx + dy*dy - di*di
                
                cons.append({'type': 'ineq', 'fun': overlap_con})
        
        return cons

    # SLSQP is good for this size, but might be slow with many constraints.
    # Let's use a penalty method approach with BFGS for robustness if SLSQP struggles,
    # but SLSQP is the standard choice. 
    # Given N=26, constraints ~ 100 (boundary) + 325 (overlap) = 425.
    # This is heavy. Let's try SLSQP. If it fails, fallback? 
    # Actually, let's just run it.
    
    # To speed up, we can use a simpler constraint formulation or penalty.
    # Let's try a penalty method with BFGS which is faster and robust for non-convex.
    
    def penalty_objective(vars):
        cx = vars[0::3]
        cy = vars[1::3]
        r = vars[2::3]
        
        # Objective: -sum(r)
        obj_val = -np.sum(r)
        
        penalty = 0.0
        penalty_weight = 1000.0
        
        # Boundary penalties
        for i in range(n):
            x, y, ri = cx[i], cy[i], r[i]
            # x < r
            if x < ri:
                penalty += penalty_weight * (ri - x)**2
            # x > 1-r
            if x > 1 - ri:
                penalty += penalty_weight * (x - (1 - ri))**2
            # y < r
            if y < ri:
                penalty += penalty_weight * (ri - y)**2
            # y > 1-r
            if y > 1 - ri:
                penalty += penalty_weight * (y - (1 - ri))**2
                
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                req_dist = r[i] + r[j]
                
                if dist < req_dist:
                    penetration = req_dist - dist
                    penalty += penalty_weight * penetration**2
                    
        return obj_val + penalty

    # Optimization
    res = minimize(penalty_objective, x0, method='BFGS', options={'maxiter': 500})
    
    final_vars = res.x
    final_centers = np.column_stack((final_vars[0::3], final_vars[1::3]))
    final_radii = final_vars[2::3]
    
    # Post-processing: Ensure strict validity
    # Sometimes optimization leaves tiny violations.
    # Clip radii to be valid.
    # If a circle is invalid, reduce its radius slightly.
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        
        # Boundary check
        max_r_boundary = min(x, 1-x, y, 1-y)
        if max_r_boundary < 0:
            max_r_boundary = 0 # Should not happen if centers in [0,1]
            
        r = min(r, max_r_boundary - 1e-9)
        
        # Overlap check
        for j in range(i + 1, n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            rj = final_radii[j]
            
            if dist < r + rj:
                # Reduce r_i to fit
                new_r = (dist - rj) / 2.0 # Share the gap? Or just ensure valid.
                # Actually, we want to keep sum high. 
                # If overlap, reduce larger one? Or just reduce i.
                # Simple fix: r_i = dist - r_j - epsilon
                valid_r = dist - rj - 1e-9
                if valid_r < r:
                    r = valid_r
                    if r < 0: r = 0
        
        final_radii[i] = r

    # Recalculate sum
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper for validation logic (not part of return, just for checking internally if needed)
# But the prompt asks to return the result.

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation check
    import math
    valid = True
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            valid = False
            break
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
            if d < radii[i] + radii[j] - 1e-9:
                valid = False
                break
    print(f"Valid: {valid}")
