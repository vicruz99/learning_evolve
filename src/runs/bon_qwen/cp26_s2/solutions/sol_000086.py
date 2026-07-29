# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 391a5ee9) state=179e45e1 sum of radii=1.671429 correctness=1.0
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
        elif np.isnan(radii[i]):
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

def run_packing():
    n = 26
    
    # Helper to convert flat vector to centers and radii
    def unpack(v):
        centers = v[:2*n].reshape(n, 2)
        radii = v[2*n:]
        return centers, radii

    # Helper to define constraints
    def get_constraints():
        constraints = []
        # Boundary constraints
        for i in range(n):
            xi = 2*i
            yi = 2*i + 1
            ri = 2*n + i
            
            # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n+i]})
            # 1 - x >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[2*n+i]})
            # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n+i]})
            # 1 - y >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[2*n+i]})
            
        # Overlap constraints: dist^2 >= (r1+r2)^2
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = 2*i, 2*i+1
                xj, yj = 2*j, 2*j+1
                ri, rj = 2*n+i, 2*n+j
                
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                def fun(v, i=i, j=j):
                    dx = v[xi] - v[xj]
                    dy = v[yi] - v[yj]
                    dr = v[ri] + v[rj]
                    return dx*dx + dy*dy - dr*dr
                constraints.append({'type': 'ineq', 'fun': fun})
        
        return constraints

    constraints = get_constraints()
    
    # Bounds: x,y in [0,1], r >= 0. 
    # Note: constraints enforce r <= 0.5 implicitly, but we can bound r <= 0.5 for safety.
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    best_sum = -1.0
    best_v = None
    
    # Try multiple initializations
    # 1. Hexagonal-like grid
    # 2. Random perturbed grid
    
    inits = []
    
    # Init 1: Grid
    # We need 26 points. 6x5 grid has 30 points.
    # Let's pick 6 cols, 5 rows, spacing to fit in [0,1].
    # x: 6 points, y: 5 points.
    x_vals = np.linspace(0.1, 0.9, 6)
    y_vals = np.linspace(0.1, 0.9, 5)
    
    grid_centers = []
    for y in y_vals:
        for x in x_vals:
            grid_centers.append([x, y])
    
    # Take first 26
    c1 = np.array(grid_centers[:26])
    r1 = np.full(n, 0.05) # Safe radius
    inits.append((c1, r1))
    
    # Init 2: Shifted hexagonal pattern
    # Rows with alternating shifts
    c2 = []
    idx = 0
    # Try to fit 6 rows? 
    # Let's just use a random shuffle of the grid to break symmetry
    rng = np.random.RandomState(42)
    shuffled_indices = rng.permutation(30)[:26]
    c2 = np.array([grid_centers[i] for i in shuffled_indices])
    r2 = np.full(n, 0.05)
    inits.append((c2, r2))

    # Init 3: Uniform random
    rng2 = np.random.RandomState(123)
    c3 = rng2.uniform(0.1, 0.9, (n, 2))
    r3 = np.full(n, 0.04)
    inits.append((c3, r3))

    # Init 4: Hexagonal lattice structure explicitly
    # 5 rows, counts 5, 5, 5, 5, 6 (staggered)
    # Let's construct it manually
    c4 = []
    r_val_init = 0.08
    # Row 0 (5 circles)
    y0 = 0.2
    for k in range(5):
        x = 0.1 + k * 0.2
        c4.append([x, y0])
    # Row 1 (5 circles, shifted)
    y1 = 0.2 + r_val_init * math.sqrt(3) # approx 0.34
    # Adjust y to fit
    y1 = 0.35
    for k in range(5):
        x = 0.2 + k * 0.2
        c4.append([x, y1])
    # Row 2 (5 circles)
    y2 = 0.55
    for k in range(5):
        x = 0.1 + k * 0.2
        c4.append([x, y2])
    # Row 3 (5 circles)
    y3 = 0.75
    for k in range(5):
        x = 0.1 + k * 0.2
        c4.append([x, y3])
    # Row 4 (6 circles? No, we have 20. Need 6 more)
    # Let's just fill the rest.
    # Current 20. Need 6.
    # Maybe a 6th row?
    y4 = 0.9
    for k in range(6):
        x = 0.05 + k * 0.15 # Squeezed
        c4.append([x, y4])
    
    c4 = np.array(c4)
    r4 = np.full(n, 0.04)
    inits.append((c4, r4))

    for centers, radii in inits:
        # Normalize centers to be within [0,1] if slightly out (though construction ensures it)
        centers = np.clip(centers, 1e-5, 1-1e-5)
        radii = np.clip(radii, 1e-6, 0.45)
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        try:
            res = minimize(
                fun=lambda v: -np.sum(v[2*n:]), # Maximize sum of radii
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success or (not np.isnan(res.fun) and np.sum(res.x[2*n:]) > best_sum):
                # Check validity of result
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt = res.x[2*n:]
                
                # Small fix: ensure non-negative radii
                r_opt = np.maximum(r_opt, 1e-9)
                
                # Validate
                if validate_packing(c_opt, r_opt):
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_v = res.x.copy()
        except Exception:
            pass

    if best_v is not None:
        centers = best_v[:2*n].reshape(n, 2)
        radii = best_v[2*n:]
        # Ensure radii are positive
        radii = np.maximum(radii, 1e-9)
        
        # Final sanity check and clamping to satisfy validation strictly
        # The optimizer might be very close to boundary.
        # Let's shrink radii slightly if needed to be safe? 
        # But we want to maximize.
        # The validation allows 1e-12 tolerance.
        
        # Just return the best valid found
        return centers, radii, np.sum(radii)
    else:
        # Fallback to a valid packing if optimization fails
        # Simple grid
        centers = np.array([[0.5, 0.5]] * n) # This will fail validation
        # Generate valid grid
        rows = 5
        cols = 6
        # 26 circles
        step_x = 1.0 / (cols + 1)
        step_y = 1.0 / (rows + 1)
        centers = []
        radii = []
        count = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if count < n:
                    centers.append([c * step_x, r * step_y])
                    # Radius roughly half min spacing
                    r_est = 0.5 * min(step_x, step_y) * 0.9
                    radii.append(r_est)
                    count += 1
        
        centers = np.array(centers)
        radii = np.array(radii)
        return centers, radii, np.sum(radii)
