# sol_000236 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e58a758a) state=bad81648 sum of radii=2.557970 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
from math import sqrt

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    # Objective: maximize sum of radii => minimize -sum(radii)
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total variables: 3 * 26 = 78
    
    # We will try to find a good packing by optimizing positions and radii.
    # Since the problem is non-convex, we will use a heuristic initial guess 
    # and then refine.
    
    # Strategy:
    # 1. Initialize circles in a dense hexagonal-like pattern.
    # 2. Use an optimizer to maximize sum of radii.
    
    # Helper to create initial guess
    def get_initial_guess():
        # Try to arrange 26 circles in a grid/hex pattern
        # A 5x5 grid has 25. We need 1 more.
        # Maybe a 6x5 grid scaled down? Or hexagonal.
        
        # Let's try a hexagonal packing approximation
        # Rows of 5, 5, 5, 5, 5, 1 ? 
        # Or 5, 6, 5, 6, 4? (Sum 26) -> 6 is too wide for r > 0.08
        # Let's try 5 rows of 5 and 1 in center?
        
        # Better: Start with a square grid of 26 points, slightly randomized
        # Or just a hexagonal grid.
        
        # Let's place them in a 6x5 grid (30 spots) and pick 26? 
        # No, let's just place 26 points.
        
        # A good packing for 26 might be a 5x5 grid with one added?
        # But 5x5 is full.
        # Let's try a lattice with spacing that fits 26.
        # Area heuristic: 26 * pi * r^2 approx 0.85 => r approx 0.101
        # Diameter approx 0.202.
        # 1/0.202 approx 4.95. So roughly 5 circles fit in width.
        # So we can have 5 columns.
        # 26/5 = 5.2 rows. So 6 rows.
        
        # Let's arrange in 6 rows.
        # Row lengths: 5, 4, 5, 4, 5, 3? Sum = 26.
        # Or 5, 5, 5, 5, 5, 1?
        
        # Let's try 6 rows with varying counts to fit hexagonal spacing.
        # Hex spacing: dx = 2r, dy = sqrt(3)r.
        # If r ~ 0.1, dx ~ 0.2, dy ~ 0.173.
        # 6 rows height: 2r + 5*dy = 0.2 + 0.866 = 1.066 > 1.
        # So we need to compress or use different arrangement.
        
        # Let's just start with a random feasible packing with small radii
        # and let optimizer grow them.
        
        centers = np.random.rand(n, 2)
        radii = np.ones(n) * 0.05 # Small radius to start
        
        return centers, radii

    # Let's try a structured initialization
    def structured_init():
        # Try to fit 26 circles in a hexagonal lattice pattern
        # We can define a lattice and select 26 points
        # Or just place them.
        
        # Let's try 5 rows of 5, and 1 extra.
        # But 5x5 grid has r=0.1.
        # To fit 26, we must reduce r.
        # Let's start with r=0.09.
        
        centers = []
        radii = []
        r = 0.09
        
        # 5 rows, 5 cols
        # Grid positions
        for i in range(5):
            for j in range(5):
                x = r + j * 2 * r
                y = r + i * 2 * r
                centers.append([x, y])
                radii.append(r)
        
        # Add 26th circle
        # Maybe in the center? (0.5, 0.5) is occupied.
        # Maybe at (0.5, 0.5 + something)
        # Or just random
        centers.append([0.5, 0.5])
        radii.append(r)
        
        # This has overlaps. We need to resolve.
        return np.array(centers), np.array(radii)

    # Optimization function
    # We want to maximize sum(radii).
    # Equivalent to minimizing -sum(radii).
    # Constraints:
    # 1. Boundary: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # 2. Non-overlap: dist(i, j) >= r_i + r_j
    
    # We can use a penalty method or constrained optimization.
    # scipy.optimize.minimize supports bounds and constraints.
    
    # Let's use bounds for x, y in [0, 1] and r >= 0.
    # But x, y, r are coupled.
    # Better to optimize x, y, r directly.
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5].
    
    # Constraints for minimize:
    # fun(x) >= 0
    
    # This might be slow. Let's try a simpler approach:
    # Fix radii to be equal?
    # If we fix radii equal to r, we just need to find positions.
    # But we want to maximize sum, so we can vary r.
    
    # Let's try to maximize sum(r) with constraints.
    # Using SLSQP method.
    
    def objective(vars_flat):
        # vars_flat shape (78,)
        # reshape to (26, 3)
        pts = vars_flat.reshape(n, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        # Return negative sum of radii
        return -np.sum(r)

    def constraint_boundary(vars_flat):
        pts = vars_flat.reshape(n, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        # x - r >= 0
        # x + r <= 1 => 1 - (x + r) >= 0
        # y - r >= 0
        # y + r <= 1 => 1 - (y + r) >= 0
        
        c = np.concatenate([
            x - r,
            1 - (x + r),
            y - r,
            1 - (y + r)
        ])
        return c

    def constraint_overlap(vars_flat):
        pts = vars_flat.reshape(n, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        c = []
        for i in range(n):
            for j in range(i + 1, n):
                # dist^2 >= (r_i + r_j)^2
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                dist_sq = dx*dx + dy*dy
                val = dist_sq - dr*dr
                c.append(val)
        return np.array(c)

    # Initial guess
    # Let's try a grid start with small radius
    centers_init = np.zeros((n, 2))
    # 5x5 grid + 1
    k = 0
    for i in range(5):
        for j in range(5):
            centers_init[k, 0] = 0.1 + j * 0.2
            centers_init[k, 1] = 0.1 + i * 0.2
            k += 1
    # Last one
    centers_init[k, 0] = 0.5
    centers_init[k, 1] = 0.5
    
    radii_init = np.ones(n) * 0.05
    
    # Flatten
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Constraints
    # SLSQP requires callable constraints
    def make_bc(vars_flat):
        return constraint_boundary(vars_flat)
    
    def make_oc(vars_flat):
        return constraint_overlap(vars_flat)

    cons = [
        {'type': 'ineq', 'fun': make_bc},
        {'type': 'ineq', 'fun': make_oc}
    ]

    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            final_vars = res.x
        else:
            # If optimization fails, return initial or best found
            final_vars = x0 # Fallback, though unlikely to be good
    except Exception as e:
        # Fallback
        final_vars = x0

    # Extract results
    final_vars = final_vars.reshape(n, 3)
    centers = final_vars[:, :2]
    radii = final_vars[:, 2]
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 1e-9)
    
    # Validate (internal check)
    # We assume the optimizer respected constraints.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
