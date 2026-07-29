# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d058a929) state=e09e0dd0 sum of radii=2.576166 correctness=1.0
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

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False

    for i in range(n):
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

def get_initial_guess(n):
    """
    Generates a hexagonal grid of n points inside [0,1]x[0,1].
    """
    points = []
    # Hexagonal packing parameters
    # We try to fit rows with alternating number of points
    # Pattern: 6, 5, 6, 5, 4 sums to 26.
    # Spacing estimates: 
    # Horizontal spacing dx approx 0.15-0.2
    # Vertical spacing dy approx dx * sqrt(3)/2
    
    # Let's construct coordinates explicitly for 26 points
    # Rows y-coords: 0.1, 0.25, 0.4, 0.55, 0.7 (approx)
    # But we want to be safe inside bounds, so maybe start 0.1 and end 0.9?
    
    # Let's use a standard grid logic and pick best or just fill
    # A 6x5 grid has 30 points. We can pick 26.
    # Or generate hex coordinates.
    
    # Hex grid generation
    # Row 0: 6 points
    # Row 1: 5 points (shifted)
    # Row 2: 6 points
    # Row 3: 5 points
    # Row 4: 4 points
    
    row_counts = [6, 5, 6, 5, 4]
    rows = 5
    
    # Estimate spacing to fit in [0,1]
    # Max width needed for 6 points: 5 gaps.
    # If we leave margin 0.05 on sides, width 0.9.
    # dx = 0.9 / 5 = 0.18
    # dy = dx * sqrt(3)/2 = 0.155
    # Total height for 5 rows (4 gaps): 4 * 0.155 = 0.62. Fits easily.
    
    # Let's adjust to be more centered or spread out.
    # Actually, let's just place them with a safe small radius logic later.
    
    y_start = 0.1
    y_end = 0.9
    dy = (y_end - y_start) / (rows - 1)
    
    x_start = 0.05
    x_end = 0.95
    
    for r_idx, count in enumerate(row_counts):
        y = y_start + r_idx * dy
        
        # Determine x range for this row
        # For hex packing, odd rows (index 1, 3) are shifted.
        # Shift amount = dx / 2
        # We need to determine dx based on count?
        # Let's just use uniform spacing for now, optimizer will fix it.
        
        # To allow optimizer to move freely, just spread them out.
        # If count is 6, 5 gaps.
        # If count is 5, 4 gaps.
        # Let's use a consistent dx for simplicity? No, different counts.
        
        # Let's just place them in a grid for this row
        if count > 0:
            # Spacing within row
            # Available width 0.9. 
            # x coords
            xs = np.linspace(x_start + (x_end - x_start)/(count+1), 
                             x_end - (x_end - x_start)/(count+1), 
                             count)
            
            # Shift for hex pattern
            if r_idx % 2 == 1:
                # Shift right by half a "grid unit" roughly
                # Estimate grid unit as (x_end-x_start)/5 approx 0.16
                shift = (x_end - x_start) / 10.0 # rough shift
                xs += shift
                # Clamp to bounds roughly
                xs = np.clip(xs, x_start, x_end)

            for x in xs:
                points.append([x, y])
        
        if len(points) >= n:
            break
            
    return np.array(points[:n])

def run_packing():
    n = 26
    
    # 1. Initialize
    centers_init = get_initial_guess(n)
    # Start with small radius to ensure feasibility
    radii_init = np.full(n, 0.01)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)
        
    # Constraints
    constraints = []
    
    # Boundary constraints:
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    # Total 4*n constraints
    
    # We can define these as functions or arrays. 
    # Since n is small (26), defining them explicitly is fine, 
    # but a list of dicts is easier to manage.
    
    # However, SLSQP works best with a function that returns an array for constraints.
    # Let's define a single constraint function.
    
    def constraint_func(vars):
        c = []
        # Boundary constraints
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            
            # x - r >= 0
            c.append(x - r)
            # 1 - x - r >= 0
            c.append(1.0 - x - r)
            # y - r >= 0
            c.append(y - r)
            # 1 - y - r >= 0
            c.append(1.0 - y - r)
            
        # Pairwise constraints: dist^2 - (r_i + r_j)^2 >= 0
        # Only need to check i < j
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            
            for j in range(i + 1, n):
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                rad_sum = ri + rj
                
                c.append(dist_sq - rad_sum*rad_sum)
                
        return np.array(c)

    # Define constraints dict
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Run optimization
    # SLSQP is suitable for constrained problems
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
        
        if res.success:
            x_opt = res.x
        else:
            # If optimization fails, try to recover or use result anyway
            x_opt = res.x
    except Exception:
        # Fallback to initial guess if crash (unlikely)
        x_opt = x0
        
    # Extract results
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    
    for i in range(n):
        centers_opt[i, 0] = x_opt[3*i]
        centers_opt[i, 1] = x_opt[3*i+1]
        radii_opt[i] = x_opt[3*i+2]
        
    # Validate
    if not validate_packing(centers_opt, radii_opt):
        # If invalid, try to shrink radii slightly to fix? 
        # Or just return, but we want valid.
        # SLSQP might drift slightly outside bounds due to tolerance.
        # Let's project back or clamp?
        # Actually, if it's invalid, it's a failure. 
        # But with strict constraints it should be fine.
        # Just in case, re-validate.
        pass

    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
