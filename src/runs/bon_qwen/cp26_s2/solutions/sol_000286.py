# sol_000286 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a4dfceb8) state=9c3aac0c sum of radii=2.577390 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def unpack_variables(params, n):
    """Converts flat parameter array to centers and radii."""
    # params is [x1, y1, r1, x2, y2, r2, ...]
    # Shape (3*n,)
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        idx = 3 * i
        centers[i, 0] = params[idx]
        centers[i, 1] = params[idx+1]
        radii[i] = params[idx+2]
    return centers, radii

def pack_circles(n=26):
    """
    Optimizes circle packing to maximize sum of radii.
    """
    # Initial guess: Hexagonal grid arrangement
    # We want to fit n circles.
    # Let's estimate rows and cols.
    # Approx area per circle ~ (2r)^2 * sqrt(3)/2 ? 
    # Let's just create a dense grid.
    
    # Heuristic: 5-6 rows
    rows = 6
    cols = 5
    # 5 rows of 5 and 6?
    # Let's generate points
    
    points = []
    r_init = 0.06 # Initial small radius to avoid overlaps
    
    # Create a hexagonal lattice pattern
    # Row 0: 5 circles
    # Row 1: 6 circles
    # Row 2: 5 circles
    # Row 3: 6 circles
    # Row 4: 5 circles
    # Row 5: ?
    # Total so far: 5+6+5+6+5 = 27. We need 26.
    # Remove one.
    
    y_step = 1.0 / 6.0
    x_step = 1.0 / 6.0
    
    # Let's just place them in a grid and shift odd rows
    # This is a rough initialization. The optimizer will fix positions.
    # We need to ensure no overlaps initially.
    # r_init = 0.05 is safe for spacing ~0.16
    
    current_n = 0
    for r in range(rows):
        # Determine number of circles in this row
        # Alternating 5 and 6
        if r % 2 == 0:
            count = 5
        else:
            count = 6
        
        for c in range(count):
            if current_n >= n:
                break
            
            # Hexagonal spacing
            # x = (c + 0.5 + (r%2)*0.5) * spacing
            # Let's map to [0,1]
            
            # Width occupied by 'count' circles with spacing s is approx count * s
            # Let's just distribute them uniformly in x
            # But with offset
            
            # Simple uniform distribution in a bounding box, then optimizer adjusts
            # To be safe from overlaps, let's use a standard grid
            # Grid spacing 0.15
            # 0.15 * 6 = 0.9
            grid_x = 0.1 + c * 0.15
            grid_y = 0.1 + r * 0.15
            
            # Add slight offset for hex packing
            grid_x += (r % 2) * 0.075
            
            # Clamp to [0,1]
            grid_x = max(0.0, min(1.0, grid_x))
            grid_y = max(0.0, min(1.0, grid_y))
            
            points.append((grid_x, grid_y))
            current_n += 1
            
    # Ensure we have n points
    while len(points) < n:
        points.append((np.random.rand(), np.random.rand()))
        
    # Trim if more
    points = points[:n]
    
    # Initial radii
    # Must be small enough to not overlap with this grid
    # Grid spacing is ~0.15. Min dist ~0.15.
    # 2*r < 0.15 => r < 0.075. Let's use 0.05.
    r_init = 0.04
    
    # Build initial params
    # Order: x1, y1, r1, x2, y2, r2...
    initial_params = []
    for x, y in points:
        initial_params.extend([x, y, r_init])
        
    initial_params = np.array(initial_params)
    
    # Bounds
    # x in [0, 1]
    # y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Constraints
    constraints = []
    
    # 1. Boundary constraints
    # x - r >= 0  => x - r >= 0
    # x + r <= 1  => 1 - x - r >= 0
    # y - r >= 0  => y - r >= 0
    # y + r <= 1  => 1 - y - r >= 0
    # These can be added as linear constraints or just handled by bounds?
    # Bounds on x,y are [0,1]. Bounds on r are [0, 0.5].
    # But x and r are coupled.
    # We can add inequality constraints:
    # c(x, r) = x - r >= 0
    # c(x, r) = 1 - x - r >= 0
    # etc.
    
    # Function to compute constraint values
    def boundary_constraints(params):
        vals = []
        for i in range(n):
            idx = 3 * i
            x = params[idx]
            y = params[idx+1]
            r = params[idx+2]
            
            # x >= r
            vals.append(x - r)
            # x + r <= 1
            vals.append(1.0 - x - r)
            # y >= r
            vals.append(y - r)
            # y + r <= 1
            vals.append(1.0 - y - r)
        return np.array(vals)

    constraints.append({
        'type': 'ineq',
        'fun': boundary_constraints
    })
    
    # 2. Non-overlap constraints
    # dist_ij >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # This is O(N^2). For N=26, ~325 constraints.
    # Vectorization might help, but loop is fine for N=26.
    
    def non_overlap_constraints(params):
        vals = []
        for i in range(n):
            idx_i = 3 * i
            xi = params[idx_i]
            yi = params[idx_i+1]
            ri = params[idx_i+2]
            
            for j in range(i + 1, n):
                idx_j = 3 * j
                xj = params[idx_j]
                yj = params[idx_j+1]
                rj = params[idx_j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                val = dist_sq - sum_r**2
                vals.append(val)
        return np.array(vals)

    constraints.append({
        'type': 'ineq',
        'fun': non_overlap_constraints
    })
    
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)
    
    # Optimize
    # SLSQP is good for this.
    # We might need to run multiple times or use a good start.
    # The grid start is decent.
    
    try:
        res = opt.minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if res.success:
            centers, radii = unpack_variables(res.x, n)
            sum_radii = np.sum(radii)
            return centers, radii, sum_radii
        else:
            # If optimization failed, return initial guess?
            # Or try to fix constraints manually?
            # Let's just return the best found even if not successful, 
            # or the initial params.
            centers, radii = unpack_variables(res.x, n)
            sum_radii = np.sum(radii)
            return centers, radii, sum_radii
            
    except Exception as e:
        # Fallback to a simple valid packing if optimizer crashes
        # Uniform grid
        centers_fallback = np.zeros((n, 2))
        radii_fallback = np.zeros(n)
        # 5x5 grid + 1
        k = 0
        for r in range(5):
            for c in range(5):
                if k >= n: break
                centers_fallback[k] = [0.1 + c*0.2, 0.1 + r*0.2]
                radii_fallback[k] = 0.05 # Safe radius
                k += 1
        while k < n:
             # Place remaining in gaps
             centers_fallback[k] = [0.5, 0.5]
             radii_fallback[k] = 0.01
             k += 1
        return centers_fallback, radii_fallback, np.sum(radii_fallback)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers, radii, sum_radii = pack_circles(26)
    return centers, radii, sum_radii

# Helper to ensure no lambda functions and top level definitions as requested
# The functions above are top level.
