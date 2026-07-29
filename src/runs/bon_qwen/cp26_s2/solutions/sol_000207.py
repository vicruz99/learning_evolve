# sol_000207 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fb167b6) state=9d7fbabe sum of radii=2.618126 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def get_constraints_func(n):
    """
    Creates a list of constraint dictionaries for scipy.optimize.minimize.
    Constraints:
    1. Boundaries: x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    2. Non-overlap: dist(i, j) >= r_i + r_j
    """
    constraints = []
    
    # Helper to create bound constraints for each circle
    # Variables layout: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
    # Indices:
    # x_i -> 2*i
    # y_i -> 2*i + 1
    # r_i -> 2*n + i
    
    for i in range(n):
        xi_idx = 2 * i
        yi_idx = 2 * i + 1
        ri_idx = 2 * n + i
        
        # x_i >= r_i  => x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, xi=xi_idx, ri=ri_idx: x[xi] - x[ri]
        })
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, xi=xi_idx, ri=ri_idx: 1.0 - x[xi] - x[ri]
        })
        # y_i >= r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, yi=yi_idx, ri=ri_idx: x[yi] - x[ri]
        })
        # y_i <= 1 - r_i
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, yi=yi_idx, ri=ri_idx: 1.0 - x[yi] - x[ri]
        })

    # Non-overlap constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            xi_idx = 2 * i
            yi_idx = 2 * i + 1
            xj_idx = 2 * j
            yj_idx = 2 * j + 1
            ri_idx = 2 * n + i
            rj_idx = 2 * n + j
            
            # dist >= r_i + r_j
            # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
            # To avoid sqrt issues with gradient, we can use squared distance if careful,
            # but sqrt is differentiable everywhere except 0, which shouldn't happen for valid packs.
            # Actually, using squared form: (xi-xj)^2 + ... >= (ri+rj)^2
            # But this is not convex and derivatives might be tricky at 0.
            # Standard distance is fine.
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi_idx, yi=yi_idx, xj=xj_idx, yj=yj_idx, ri=ri_idx, rj=rj_idx: 
                    np.sqrt((x[xi] - x[xj])**2 + (x[yi] - x[yj])**2) - (x[ri] + x[rj])
            })
            
    return constraints

def objective_function(x, n):
    """
    Maximize sum of radii. Minimize negative sum.
    Variables: [x0, y0, ..., x25, y25, r0, ..., r25]
    """
    radii = x[2*n:]
    return -np.sum(radii)

def generate_initial_guess(n, strategy='grid'):
    """
    Generates an initial guess for centers and radii.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if strategy == 'grid':
        # 5x5 grid covers 25, add 1
        # Grid points
        pts = []
        # 5 points in [0,1] evenly spaced with margin
        # Actually, let's just put them in center
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        for y in ys:
            for x in xs:
                if len(pts) < n:
                    pts.append([x, y])
        
        # If we need more (though 5x5=25), fill remaining
        # For 26, we have 1 left. Place in a gap.
        # Center of square is (0.5, 0.5). If occupied, try (0.2, 0.2)
        if n > 25:
            pts.append([0.2, 0.2])
            
        centers = np.array(pts[:n])
        
    elif strategy == 'hexagonal':
        # Hexagonal packing layout
        row_radius = 0.08 # small initial radius
        row_y = row_radius
        count = 0
        row_idx = 0
        while count < n:
            # Number of circles in this row
            # Alternating rows might have different counts or shifts
            # Simple hex grid: shift by r every other row
            # Let's estimate how many fit in width 1
            # width approx 1. radius 0.08. diameter 0.16.
            # 1/0.16 approx 6.
            n_in_row = 6 if row_idx % 2 == 0 else 5
            
            # Adjust if this pushes us over n
            needed = n - count
            n_in_row = min(n_in_row, needed)
            
            if n_in_row <= 0:
                break
                
            # X coordinates
            if row_idx % 2 == 0:
                # Even row: start at r, step 2r
                xs = np.linspace(row_radius, 1 - row_radius, n_in_row)
            else:
                # Odd row: start at 2r (shifted by r), step 2r
                # Actually shift by r means start at 2r?
                # Center of gap between 0 and 2r is r.
                # Wait, standard hex:
                # Row 0: r, 3r, 5r...
                # Row 1: 2r, 4r, 6r...
                # So start at 2r.
                # Check bounds: last at 2r + (k-1)2r.
                # Let's just generate evenly spaced but shifted
                # Range [2r, 1-r]
                start = 2 * row_radius
                end = 1 - row_radius
                if n_in_row == 1:
                    xs = np.array([(start + end) / 2])
                else:
                    xs = np.linspace(start, end, n_in_row)
            
            centers[count:count+n_in_row, 0] = xs
            centers[count:count+n_in_row, 1] = row_y
            count += n_in_row
            
            # Increment Y for next row. Hex spacing is r * sqrt(3)
            row_y += row_radius * np.sqrt(3)
            row_idx += 1
            
            # Stop if Y exceeds boundary
            if row_y + row_radius > 1.0:
                break
                
    # Set initial radii
    radii[:] = 0.05
    
    return centers, radii

def run_packing():
    n = 26
    num_vars = 2 * n + n  # x, y, r for each circle
    
    best_result = None
    best_sum = -np.inf
    
    # Try multiple initializations
    strategies = ['grid', 'hexagonal']
    
    # Also try random restarts
    for strat in strategies:
        # Run a few times with slight noise or different configs if needed
        # But 'grid' and 'hexagonal' are distinct enough.
        # Let's run 3 times for each to be safe? 
        # Time complexity might be an issue, but N=26 is small.
        for _ in range(2):
            try:
                centers, radii = generate_initial_guess(n, strategy=strat)
                
                # Add small random noise to break symmetry
                centers += np.random.randn(n, 2) * 0.001
                radii += np.random.rand(n) * 0.001
                
                # Flatten to 1D array
                x0 = np.hstack([centers.flatten(), radii])
                
                # Bounds
                # x, y in [0, 1]
                # r in [0, 0.5] (cannot be larger than 0.5)
                bounds = []
                for i in range(2 * n):
                    bounds.append((0.0, 1.0))
                for i in range(n):
                    bounds.append((0.0, 0.5))
                
                constraints = get_constraints_func(n)
                
                # Optimize
                # SLSQP is good for constrained problems
                res = minimize(
                    objective_function,
                    x0,
                    args=(n,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
                )
                
                if res.success or (not np.isinf(res.fun) and not np.isnan(res.fun)):
                    current_sum = -res.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = res
                        
            except Exception as e:
                # If optimization fails, continue
                continue

    if best_result is None:
        # Fallback: return a simple valid packing (small circles)
        centers, radii = generate_initial_guess(n, strategy='grid')
        radii[:] = 0.01
        return centers, radii, np.sum(radii)

    # Extract best solution
    x_opt = best_result.x
    centers = x_opt[:2*n].reshape((n, 2))
    radii = x_opt[2*n:]
    
    # Validate and fix small numerical errors
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    # Ensure centers are within bounds relative to radii (clamp if necessary)
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        # Clamp x
        centers[i, 0] = np.clip(x, r, 1.0 - r)
        # Clamp y
        centers[i, 1] = np.clip(y, r, 1.0 - r)
        
    # Note: Clamping might cause overlaps, but the optimizer should have found a valid point.
    # The clamping is just for safety against float errors.
    
    return centers, radii, np.sum(radii)
