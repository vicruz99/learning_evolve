# sol_000264 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=203e1cc1 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def objective(x, n):
    """
    Objective function to minimize.
    We want to maximize sum of radii. Assuming equal radii r, sum = n * r.
    So we minimize -n * r.
    x contains [x1, y1, ..., xn, yn, r]
    """
    r = x[-1]
    return -r * n

def constraints(x, n):
    """
    Constraint function for boundary and non-overlap conditions.
    Returns a list of constraint values >= 0.
    x contains [x1, y1, ..., xn, yn, r]
    """
    r = x[-1]
    c = x[:2*n].reshape(n, 2)
    res = []
    
    # Boundary constraints:
    # For each circle, center must be at least r away from boundaries.
    # x >= r  => x - r >= 0
    # x <= 1-r => r - (x - 1) >= 0
    # Similarly for y.
    for i in range(n):
        res.append(c[i,0] - r)
        res.append(r - (c[i,0] - 1))
        res.append(c[i,1] - r)
        res.append(r - (c[i,1] - 1))
        
    # Overlap constraints:
    # Distance between centers must be >= 2r.
    # Squared distance >= (2r)^2 = 4r^2.
    # ||ci - cj||^2 - 4r^2 >= 0
    for i in range(n):
        ci = c[i]
        for j in range(i+1, n):
            cj = c[j]
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            res.append(dx*dx + dy*dy - 4*r*r)
            
    return res

def run_packing():
    n = 26
    
    # 1. Generate initial hexagonal packing configuration
    # We use a pattern of rows: 5, 4, 5, 4, 5, 3 circles.
    # This sums to 26 and mimics a dense hexagonal lattice.
    rows_config = [5, 4, 5, 4, 5, 3]
    points = []
    current_y = 0
    row_idx = 0
    
    for count in rows_config:
        # Hexagonal packing: alternate rows are shifted horizontally by 1 unit (if spacing is 2)
        offset = 0 if row_idx % 2 == 0 else 1
        for i in range(count):
            x = offset + 2 * i
            y = current_y
            points.append([x, y])
        current_y += math.sqrt(3) # Vertical spacing for hex lattice with dist 2
        row_idx += 1
        
    pts = np.array(points)
    
    # 2. Normalize initial configuration to fit in unit square [0, 1]x[0, 1]
    min_coords = pts.min(axis=0)
    max_coords = pts.max(axis=0)
    pts -= min_coords
    
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]
    
    # Scale to fit with a small margin (0.95) to allow optimizer to expand
    scale = 0.95 / max(width, height)
    pts *= scale
    
    # 3. Estimate initial radius based on minimum distance
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
        # Check distance to boundaries
        d_b = min(pts[i,0], 1-pts[i,0], pts[i,1], 1-pts[i,1])
        if d_b < min_dist:
            min_dist = d_b
            
    r_init = min_dist / 2.0
    
    # 4. Setup optimization variables
    # Variables: [x1, y1, ..., x26, y26, r]
    x0 = np.concatenate([pts.flatten(), [r_init]])
    
    # Bounds: coordinates in [0, 1], radius in [0, 0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)]
    
    # 5. Run optimization
    result = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                      options={'maxiter': 2000, 'ftol': 1e-12})
    
    if result.success:
        r_opt = result.x[-1]
        c_opt = result.x[:2*n].reshape(n, 2)
        
        # 6. Verification and safety clamp
        # Recompute minimum distance to ensure validity against numerical errors
        min_d = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c_opt[i] - c_opt[j])
                if d < min_d:
                    min_d = d
            d_b = min(c_opt[i,0], 1-c_opt[i,0], c_opt[i,1], 1-c_opt[i,1])
            if d_b < min_d:
                min_d = d_b
        
        # Radius cannot exceed half the minimum separation
        safe_r = min_d / 2.0
        if safe_r < r_opt - 1e-6:
            r_opt = safe_r
            
        # Construct radii array
        radii = np.full(n, r_opt)
        sum_radii = 26 * r_opt
        
        return c_opt, radii, sum_radii
    else:
        # Fallback to initial configuration if optimization fails
        return pts, np.full(n, r_init), 26 * r_init
