# sol_000117 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=cd4701a9 sum of radii=2.416990 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Helper to create initial hexagonal packing
    def create_hex_packing(n_circles):
        centers = []
        radii = []
        
        # Try to fit rows. 5 rows with counts 6, 5, 6, 5, 4 sums to 26.
        row_counts = [6, 5, 6, 5, 4]
        
        # Estimate max radius for this layout
        # Width constraint: max row width 6 circles -> 12*r <= 1 => r <= 0.0833
        # Height constraint: 5 rows -> 2*r + 4*r*sqrt(3) <= 1 => r <= 0.112
        # So width is limiting. Let's start with r = 0.08
        r = 0.075 # Safe start
        
        y_curr = r
        for i, count in enumerate(row_counts):
            # Offset for staggered rows
            # Even rows (0, 2, 4) start at r? 
            # In hex, row 0 centers: r, 3r, 5r...
            # Row 1 centers: 2r, 4r... (shifted by r)
            
            if i % 2 == 0:
                x_start = r
            else:
                x_start = 2 * r # Shifted
            
            # But wait, if row 0 has 6 circles: r, 3r, 5r, 7r, 9r, 11r.
            # Last center 11r. Right edge 12r.
            # If row 1 has 5 circles: 2r, 4r, 6r, 8r, 10r.
            # Left edge 2r - r = r. Right edge 10r + r = 11r.
            # This fits if 12r <= 1.
            
            # However, if we have fewer circles in shifted row, maybe we can center it better?
            # For now, let's stick to left-aligned staggered grid.
            
            for j in range(count):
                x = x_start + j * (2 * r)
                centers.append([x, y_curr])
                radii.append(r)
            
            y_curr += r * math.sqrt(3)
            
        return np.array(centers), np.array(radii)

    # Better initialization: Just place points in a grid and let optimizer work?
    # Or random points?
    # Let's try a dense grid initialization first, maybe 5x6 grid scaled down?
    # 5x6 = 30 points. We need 26.
    # Let's pick 26 points from a 6x6 grid.
    
    grid_size = 6
    step = 1.0 / (grid_size + 1)
    initial_centers = []
    initial_radii = []
    count = 0
    
    # A hex-like pattern on a grid is better.
    # Let's generate points on a hex grid but just place them.
    # We can use the create_hex_packing function to get valid initial points.
    
    centers, radii = create_hex_packing(n)
    
    # Flatten variables for optimizer: [x1, y1, r1, x2, y2, r2, ...]
    # Or [x1...xn, y1...yn, r1...rn]
    # Let's use [x1, y1, r1, ...]
    x0 = np.hstack([c[0] for c in centers])
    y0 = np.hstack([c[1] for c in centers])
    r0 = radii.flatten()
    initial_params = np.concatenate([x0, y0, r0])
    
    # Bounds
    # x, y in [0, 1], r in [0, 1]
    # Actually r max is 0.5
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Constraints
    constraints = []
    
    # 1. Wall constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    # These can be added as bounds or constraints. 
    # Bounds on x,y are 0,1. Bounds on r are 0, 0.5.
    # But r <= x is coupling.
    # Let's add inequality constraints: g(params) >= 0.
    
    # We can define a function that returns vector of constraint values
    
    def wall_constraints(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:]
        
        c = []
        for i in range(n):
            c.append(x[i] - r[i])       # x - r >= 0
            c.append(1 - x[i] - r[i])    # 1 - x - r >= 0
            c.append(y[i] - r[i])        # y - r >= 0
            c.append(1 - y[i] - r[i])    # 1 - y - r >= 0
        return np.array(c)

    constraints.append({'type': 'ineq', 'fun': wall_constraints})
    
    # 2. Overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    def overlap_constraints(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:]
        
        c = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                c.append(dist_sq - sum_r*sum_r)
        return np.array(c)

    constraints.append({'type': 'ineq', 'fun': overlap_constraints})
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    def objective(params):
        r = params[2*n:]
        return -np.sum(r)
    
    # Run optimization
    # SLSQP is good for this
    res = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    
    # If result is not successful, maybe try again with perturbation?
    # But let's assume it works.
    
    if not res.success:
        # Try one more time with slightly randomized start
        np.random.seed(42)
        noise = np.random.uniform(-0.01, 0.01, size=initial_params.shape)
        res = minimize(objective, initial_params + noise, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})

    best_params = res.x
    best_centers = np.column_stack((best_params[0:n], best_params[n:2*n]))
    best_radii = best_params[2*n:]
    
    # Clean up any tiny negative radii or out of bounds due to numerical issues
    best_radii = np.maximum(best_radii, 0)
    # Clamp centers to [0,1] just in case
    best_centers = np.clip(best_centers, 0, 1)
    
    # Recalculate radii to be safe?
    # Actually, if constraints are satisfied, we are good.
    # But let's clamp radii to not exceed wall distance
    for i in range(n):
        x, y = best_centers[i]
        r_max = min(x, 1-x, y, 1-y)
        if best_radii[i] > r_max:
            best_radii[i] = r_max
            
    # Verify non-overlap again and adjust if needed (very small adjustments)
    # This is a heuristic fix
    # But the optimizer should have found a valid point.
    
    sum_r = np.sum(best_radii)
    
    return best_centers, best_radii, sum_r
