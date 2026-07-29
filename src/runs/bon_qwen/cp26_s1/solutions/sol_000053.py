# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 50e7db78) state=83ddfedb sum of radii=2.596943 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Helper to compute distance squared
    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    # 1. Initialization: Hexagonal Grid
    # We try to fit 26 circles. A 5x6 grid has 30 spots. 
    # We can pick a subset or arrange in rows.
    # Let's try a pattern that mimics dense packing.
    # Rows with counts: 5, 6, 5, 6, 4? Sum = 26.
    # Or 6, 5, 6, 5, 4?
    # Let's just place them in a 6x5 grid and trim, or generate specifically.
    
    centers = []
    # Approximate spacing for r ~ 0.1 is dx=0.2, dy=0.1732
    # Let's use a simple grid first and let optimizer do the work, 
    # but perturbed to avoid symmetry issues.
    
    # Grid 6 columns, 5 rows = 30 points.
    # We take the first 26.
    # Actually, 5 rows of 5 is 25. We need 26.
    # Let's make a 6x5 grid but remove some? 
    # Better: 5 rows. Row lengths: 5, 5, 5, 5, 6?
    # Width for 6 circles with r=0.1 is 1.2 (too wide).
    # So we need to pack tighter or use hex shift.
    
    # Let's generate centers based on a hexagonal lattice with r_guess = 0.09
    r_guess = 0.09
    dx = 2 * r_guess
    dy = math.sqrt(3) * r_guess
    
    # We want to fit 26. 
    # Let's try to fill rows.
    # Row 0: 6 circles
    # Row 1: 5 circles
    # Row 2: 6 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Total 26.
    
    rows_config = [6, 5, 6, 5, 4]
    
    current_centers = []
    for r_idx, count in enumerate(rows_config):
        y = r_guess + r_idx * dy
        # Shift odd rows by r_guess (half of dx)
        x_start = r_guess + (r_idx % 2) * r_guess
        
        # We need to center the row in the square to allow growth
        # The row width for 'count' circles is (count-1)*dx + 2*r_guess
        # Actually extent is from x_start - r_guess to x_last + r_guess
        # x_last = x_start + (count-1)*dx
        # Width = x_last - x_start + 2*r_guess = (count-1)*dx + 2*r_guess = count * dx
        # We want this to be <= 1. 
        # With r_guess=0.09, dx=0.18. 
        # 6 circles -> width 1.08 (too wide).
        # 5 circles -> width 0.90 (ok).
        # 4 circles -> width 0.72 (ok).
        
        # Adjust x_start to center the row if possible, or just align left
        # Let's just place them and let optimizer fix.
        
        for c_idx in range(count):
            x = x_start + c_idx * dx
            current_centers.append([x, y])
            
    # If we generated more or less, adjust.
    # My config sums to 26.
    centers = np.array(current_centers)
    
    # Initial radii
    radii = np.full(n, r_guess)
    
    # 2. Simulation / Refinement
    # We will use a simple repulsion + growth heuristic to get close to optimal
    
    # Normalize centers to be roughly inside [0,1]
    # Current max y: 4*dy + r + r = 4*0.155 + 0.18 = 0.8. OK.
    # Current max x: for 6 circles, 5*0.18 + 0.18 = 1.08. Might be slightly out.
    # Scale down if necessary
    max_c = np.max(centers, axis=0)
    min_c = np.min(centers, axis=0)
    # Centering
    centers -= (max_c + min_c) / 2
    centers += 0.5
    # Scale to fit with some margin
    scale = 0.8 / np.max(np.abs(centers - 0.5))
    centers = 0.5 + (centers - 0.5) * scale
    radii *= scale # Radii scale too? No, radii are variables. 
    # But initial radii should be small enough to fit.
    radii = np.full(n, 0.05) # Start small
    
    # Simulation parameters
    growth_step = 0.001
    repulsion_strength = 0.5
    max_iter = 500
    
    for _ in range(max_iter):
        # Grow radii
        # Find min distance to any constraint
        min_dist = 1.0
        violated = False
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Distance to boundaries
            d_b = min(x - r, 1 - (x + r), y - r, 1 - (y + r))
            if d_b < 0:
                violated = True
                min_dist = min(min_dist, 0) # Actually needs repulsion
            else:
                min_dist = min(min_dist, d_b)
        
        if violated:
            # Repel from boundaries
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < 0: centers[i, 0] += repulsion_strength * (-(x-r))
                if x + r > 1: centers[i, 0] -= repulsion_strength * (x+r-1)
                if y - r < 0: centers[i, 1] += repulsion_strength * (-(y-r))
                if y + r > 1: centers[i, 1] -= repulsion_strength * (y+r-1)
            continue

        # Check overlaps
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(dist_sq(centers[i], centers[j]))
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    overlap_found = True
                    # Repel
                    vec = centers[i] - centers[j]
                    dist = np.linalg.norm(vec)
                    if dist > 1e-9:
                        force = repulsion_strength * (req_dist - dist) / dist
                        centers[i] += vec * force
                        centers[j] -= vec * force
                    else:
                        # Random push
                        centers[i] += np.random.randn(2) * 0.01
                        centers[j] -= np.random.randn(2) * 0.01
        
        if not overlap_found:
            # If no overlap, try to grow
            # Grow all radii slightly
            radii += growth_step
            # Cap radii to avoid explosion if stuck (unlikely with repulsion)
            # Actually, if we grow, we might cause overlap next iter.
            # But we need to keep growing.
            # To ensure progress, we only grow if we didn't resolve overlaps in previous steps?
            # Simple logic: grow every step. If overlap, repel.
        
        # Reduce growth step over time to settle
        if _ % 100 == 0 and growth_step > 0.0001:
            growth_step *= 0.9

    # Clip centers to valid range to ensure validity before optimization
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
        
    # 3. Local Optimization with SciPy
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # We reshape centers and radii into a flat array
    x0 = np.hstack([centers.flatten(), radii])
    
    # Objective: maximize sum of radii -> minimize negative sum
    def objective(vars_flat):
        r = vars_flat[2::3] # radii are every 3rd element starting index 2
        return -np.sum(r)
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    # x - r >= 0  => x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    for i in range(n):
        idx = i * 3
        xi = idx + 0
        yi = idx + 1
        ri = idx + 2
        
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3] - v[i*3+2]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3] - v[i*3+2]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i*3+1] - v[i*3+2]})
        
    # Overlap constraints
    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # => (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = i * 3
            idx_j = j * 3
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    (v[i*3] - v[j*3])**2 + (v[i*3+1] - v[j*3+1])**2 - (v[i*3+2] + v[j*3+2])**2
            })
            
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Run optimization
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        if result.success:
            final_vars = result.x
            centers = final_vars.reshape(n, 3)[:, :2]
            radii = final_vars.reshape(n, 3)[:, 2]
        else:
            # Fallback to simulation result if optimizer fails
            pass
    except Exception:
        pass

    # Ensure valid output
    # Re-check and clip if necessary (though optimizer should handle it)
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
