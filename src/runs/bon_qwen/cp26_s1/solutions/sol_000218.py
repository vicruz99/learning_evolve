# sol_000218 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b6844e7) state=8db4edd2 sum of radii=2.569829 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    # Number of circles
    n = 26
    
    # Initial configuration: Hexagonal packing
    # Rows: 6, 5, 6, 5, 4
    rows = [6, 5, 6, 5, 4]
    points = []
    y = 0
    row_idx = 0
    for count in rows:
        # For hexagonal packing, odd rows (0-indexed) are shifted by 1 unit (in 2r scale)
        # Let's use a spacing of 2 for diameter.
        x_start = 1 if (row_idx % 2 == 1) else 0
        for i in range(count):
            x = x_start + 2 * i
            points.append((x, y))
        y += np.sqrt(3)
        row_idx += 1
    
    points = np.array(points)
    
    # Normalize points to fit in unit square with some initial radius
    # Find bounding box
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to scale such that diameter fits.
    # Let's assume a target radius r0.
    # If r0 = 0.1, diameter = 0.2.
    # Current grid unit is 2 (diameter).
    # Scale factor s = 0.2 / 2 = 0.1.
    # But we need to center it.
    
    # Let's just set initial r to 0.08 and scale points accordingly.
    r0 = 0.08
    scale = 2 * r0 / 2 # ratio of diameter to grid spacing
    
    centers = points * scale
    centers -= (np.min(centers, axis=0) - r0) # Shift so left/bottom margin is r0
    # But this might push it out of 1.
    # Better: Center the cluster in [0,1]x[0,1].
    centers -= np.min(centers, axis=0)
    centers /= np.max(centers, axis=0) # Normalize to [0,1] box of centers
    # Then we need to ensure margins.
    # If centers are in [0,1], and we want radius r, we need to shrink by 2r?
    # No, if centers are in [0,1], and r > 0, they will be out.
    # We need centers in [r, 1-r].
    
    # Let's reset.
    # Use the raw hex grid points.
    # Scale them to fit in a box of size (1-2*r0) x (1-2*r0).
    # Then add r0 to coords.
    
    target_box_size = 1 - 2 * r0
    scale = target_box_size / max(width, height)
    
    centers = points * scale
    centers -= np.min(centers, axis=0) # Move to 0
    centers += r0 # Add margin
    
    radii = np.full(n, r0)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    # Constraints
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    # We can add these as bounds on x, y if we fix r? No, r is variable.
    # So nonlinear constraints.
    
    # To simplify, we can enforce x >= r and x <= 1-r.
    # But bounds are constant.
    # So we use nonlinear constraints.
    
    for i in range(n):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]
        })
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) - v[3*i+2] - v[3*j+2]
            })
            
    # Objective: Maximize sum(r) => Minimize -sum(r)
    def objective(v):
        return -np.sum(v[2::3])
        
    # Run optimization
    # Use SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    if res.success:
        centers_opt = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
        radii_opt = res.x[2::3]
        sum_radii = np.sum(radii_opt)
    else:
        # Fallback to initial
        centers_opt = centers
        radii_opt = radii
        sum_radii = np.sum(radii)
        
    # Validate and clamp if necessary
    # The optimizer might violate constraints slightly due to tolerance.
    # We can do a simple projection if needed, but SLSQP usually respects them.
    
    return centers_opt, radii_opt, sum_radii
