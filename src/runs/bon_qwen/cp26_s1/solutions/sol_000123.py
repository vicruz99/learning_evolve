# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=12aec579 sum of radii=2.486545 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

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
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-7:
                return False
    return True

def run_packing():
    n = 26
    
    # Strategy: Optimize for equal radii first, as it's often optimal for sum of radii
    # We will try to find a configuration of 26 circles with radius r that fits in the square.
    # We maximize r.
    
    # Initial guess: Hexagonal packing
    # Approximate radius for 26 circles. 
    # Area of 26 circles <= 1. 26 * pi * r^2 <= 1 => r <= sqrt(1/(26*pi)) ~ 0.11
    # But packing density ~ 0.9, so r ~ 0.105.
    # However, boundary constraints are tight.
    
    # Let's try a grid-based initialization that is dense.
    # 5 rows. 
    # To fit 26 circles, maybe 5, 5, 5, 5, 6? No, 6 circles width > 1.
    # Maybe 5, 5, 5, 5, 5, 1?
    # Or a rotated lattice?
    
    # Let's generate a hexagonal lattice that covers the square and pick 26 points.
    # Or just place them randomly and let optimizer fix it.
    
    # Better initialization:
    # Place circles in a pattern. 
    # Try to fit 5x5 grid (25 circles) and squeeze the 26th?
    # Or try a 6x5 lattice subset.
    
    # Let's use a heuristic to place centers initially.
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09) # Start slightly smaller than 0.1 to allow movement
    
    # Fill in a hexagonal pattern
    # Row height = sqrt(3)/2 * 2r = sqrt(3) * r
    # Let's fix r_temp = 0.1 for initialization layout
    r_temp = 0.1
    dy = math.sqrt(3) * r_temp
    dx = 2 * r_temp
    
    idx = 0
    row = 0
    y = r_temp
    while idx < n:
        # Shift every other row
        x_start = r_temp + (row % 2) * r_temp
        x = x_start
        while x + r_temp <= 1.0 - 1e-9 and idx < n:
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            x += dx
        row += 1
        y += dy
        # If y goes out of bounds, reset y and try to pack tighter or just break
        if y + r_temp > 1.0 - 1e-9:
            # Wrap around or just stop? 
            # If we can't fit all, we might need to reduce r_temp or use different layout.
            # But for initialization, let's just place them.
            # A simple 5x5 grid + 1 in corner might be better init.
            break
    
    # If we didn't place all, fallback to simple grid
    if idx < n:
        # Simple grid fallback
        # Try to place in 6 rows of 5?
        # 6 * 0.2 = 1.2 > 1. 
        # Let's just randomize positions in valid range
        np.random.seed(42)
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.05)

    # Optimization
    # We want to maximize sum of radii.
    # Variables: centers (n, 2) and radii (n). Total 3n variables.
    # But equal radii assumption reduces to n*2 + 1 variables.
    # Let's try optimizing equal radii first.
    
    def objective_flat(params):
        # params: x1, y1, ..., xn, yn, r
        # Maximize -r (minimize r)
        r = params[-1]
        return -r

    def constraints_flat(params):
        # Returns list of constraint dictionaries for scipy
        # Constraints:
        # 1. r <= x_i <= 1-r
        # 2. r <= y_i <= 1-r
        # 3. dist(i, j) >= 2r
        
        r = params[-1]
        cs = params[:-1].reshape(-1, 2)
        
        cons = []
        
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[-1]}) # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - (p[2*i] - 1)}) # 1 - x - r >= 0 => r - x + 1 >= 0 => 1 - (x+r) >= 0?
            # x + r <= 1 => 1 - x - r >= 0.
            # p[2*i] is x. p[-1] is r.
            # 1 - p[2*i] - p[-1] >= 0
            
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i] - p[-1]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1] - p[-1]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i+1] - p[-1]})
            
        # Overlap constraints
        # dist^2 >= (2r)^2 => dist^2 - 4r^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                # indices in params
                # x_i = 2*i, y_i = 2*i+1
                # x_j = 2*j, y_j = 2*j+1
                # r = -1
                
                def make_cons(i, j):
                    return {'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                            ((p[2*i] - p[2*j])**2 + (p[2*i+1] - p[2*j+1])**2) - 4*(p[-1])**2}
                cons.append(make_cons(i, j))
                
        return cons

    # Flatten initial centers and r
    x0 = np.concatenate([centers.flatten(), [0.1]])
    
    # Bounds
    # x, y in [0, 1]. r in [0, 0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)]
    
    # Try to optimize
    # Note: scipy minimize with constraints might be slow or get stuck.
    # Let's try SLSQP.
    
    try:
        res = minimize(objective_flat, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints_flat(x0), # constraints depend on structure not values mostly, but lambda closures need care
                       options={'ftol': 1e-9, 'maxiter': 1000})
        
        if res.success:
            centers_opt = res.x[:-1].reshape(-1, 2)
            r_opt = res.x[-1]
            radii_opt = np.full(n, r_opt)
            
            # Check validity
            if validate_packing(centers_opt, radii_opt):
                return centers_opt, radii_opt, np.sum(radii_opt)
    except Exception as e:
        pass

    # Fallback: If optimization fails, try a hardcoded good configuration or simple heuristic.
    # Let's try a specific configuration for 26 circles.
    # A known good packing for 26 circles is likely a perturbed grid.
    # Let's try to construct a 5x5 grid and perturb.
    
    # Actually, let's try a different approach: 
    # Use a simple "expansion" algorithm without scipy to be safe and robust.
    
    # Initialize centers in a dense pattern
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.01)
    
    # Place in a grid
    # 5x5 grid centers
    grid_step = 0.2
    cx, cy = 0.1, 0.1
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < n:
                centers[idx] = [cx + c * grid_step, cy + r * grid_step]
                idx += 1
    # Place the 26th circle in a gap
    if idx < n:
        centers[idx] = [0.5, 0.5] # Center
        
    # Now expand radii and resolve collisions
    # Simple iterative repulsion
    for step in range(1000):
        # Try to increase radii
        current_r = radii[0]
        target_r = current_r * 1.001
        
        # Move centers to resolve overlap and boundary violations
        for k in range(10): # Iterations per step
            for i in range(n):
                # Boundary correction
                x, y = centers[i]
                r = radii[i]
                if x - r < 0: x = r
                if x + r > 1: x = 1 - r
                if y - r < 0: y = r
                if y + r > 1: y = 1 - r
                centers[i] = [x, y]
                
                # Collision resolution
                for j in range(n):
                    if i == j: continue
                    xi, yi = centers[i]
                    xj, yj = centers[j]
                    ri, rj = radii[i], radii[j]
                    
                    dx = xj - xi
                    dy = yj - yi
                    dist = math.hypot(dx, dy)
                    min_dist = ri + rj
                    
                    if dist < min_dist and dist > 1e-9:
                        # Push apart
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Move i away from j
                        centers[i][0] -= nx * overlap * 0.5
                        centers[i][1] -= ny * overlap * 0.5
                        # Move j away from i (optional, but helps)
                        # Actually, just moving i might be enough if we iterate all i
            
        # Update radii
        valid = True
        for i in range(n):
            radii[i] = target_r
            # Check if this radius is valid (approx)
            # We do a quick check
            x, y = centers[i]
            if x < radii[i] or x > 1 - radii[i] or y < radii[i] or y > 1 - radii[i]:
                valid = False
                break
            for j in range(i + 1, n):
                dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if dist < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Rollback radii
            for i in range(n):
                radii[i] = current_r
            break
            
    # Return result
    return centers, radii, np.sum(radii)
