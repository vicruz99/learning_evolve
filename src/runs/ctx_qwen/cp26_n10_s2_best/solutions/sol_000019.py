# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b9ac6cc) state=96e1f649 sum of radii=2.568034 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # Helper to compute distance
    def dist(c1, c2):
        return np.sqrt(np.sum((c1 - c2)**2))

    # Initial configuration: 5x5 grid plus one extra point
    # 5x5 grid has points at 0.1, 0.3, 0.5, 0.7, 0.9
    # This fits 25 circles of radius 0.1.
    # We need 26. Let's perturb this to allow growth.
    
    # Generate a base grid
    grid_coords = []
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in xs:
        for y in ys:
            grid_coords.append([x, y])
            
    # We have 25 points. We need 26.
    # Add a point in the center of a gap? 
    # Actually, 0.5 is already there. 
    # Let's just add a point at (0.5, 0.2) or something, or perturb the grid.
    # Better: Use a hexagonal packing approximation or just random shuffle + slight jitter.
    
    # Let's create a configuration that is slightly tighter than grid to force optimization to work
    # Or just random valid start.
    
    # Let's try a specific heuristic layout for 26.
    # Maybe 6 rows? 
    # Rows of 5, 5, 5, 5, 4, 2?
    # Let's just use the grid and add one point at (0.5, 0.05) - close to wall?
    # No, let's use a randomized grid with some jitter.
    
    centers_init = np.array(grid_coords, dtype=float)
    # Add 26th point. Let's put it at (0.5, 0.5) is occupied.
    # Put it at (0.2, 0.2) is occupied.
    # Let's put it at (0.1, 0.05)? No.
    # Let's just insert a point at (0.5, 0.25) roughly.
    # Actually, simpler: Randomly place 26 points in [0.05, 0.95]
    np.random.seed(42)
    centers_init = np.random.uniform(0.05, 0.95, size=(26, 2))
    
    # Initial radii small enough to be valid
    radii_init = np.full(26, 0.02)
    
    # Combine into variable vector
    # x1, y1, r1, x2, y2, r2, ...
    x0 = np.zeros(26 * 3)
    for i in range(26):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square)
    bounds = []
    for i in range(26):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints
    constraints = []
    
    # 1. Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    for i in range(26):
        idx = 3*i
        
        # x - r >= 0  => x - r
        constraints.append({
            'type': 'ineq',
            'fun': (lambda v, i=i: v[3*i] - v[3*i+2])
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': (lambda v, i=i: v[3*i+1] - v[3*i+2])
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])
        })

    # 2. Non-overlap constraints: dist(i, j) >= ri + rj
    # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
    
    for i in range(26):
        for j in range(i + 1, 26):
            idx_i = 3*i
            idx_j = 3*j
            
            def make_constraint(i, j):
                def constraint(v):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - (ri + rj)
                return constraint

            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(i, j)
            })
            
    # Objective function: Maximize sum of radii => Minimize negative sum
    def objective(v):
        total_r = 0.0
        for i in range(26):
            total_r += v[3*i+2]
        return -total_r

    # Run optimization
    # SLSQP is a good choice for constrained non-linear optimization
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                          options={'ftol': 1e-8, 'maxiter': 1000, 'disp': False})
    
    # Extract results
    final_centers = np.zeros((26, 2))
    final_radii = np.zeros(26)
    
    for i in range(26):
        final_centers[i, 0] = result.x[3*i]
        final_centers[i, 1] = result.x[3*i+1]
        final_radii[i] = result.x[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # Just in case of numerical issues, clip radii slightly if needed, but constraints should hold.
    # Also ensure radii are non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, sum_radii
