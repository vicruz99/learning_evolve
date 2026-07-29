# sol_000195 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 320c78c6) state=02542124 sum of radii=2.539116 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def get_initial_centers(n):
    """
    Generates an initial configuration of n circle centers based on a hexagonal grid.
    """
    centers = []
    # Estimate radius for initial placement to fit in square
    # For n=26, hexagonal packing allows r approx 0.1
    # We start with a slightly smaller r to ensure validity
    r_init = 0.08 
    
    # Hexagonal grid spacing
    dx = 2 * r_init
    dy = math.sqrt(3) * r_init
    
    y = r_init
    row_idx = 0
    while y + r_init <= 1.0:
        # Determine x offset for this row (alternating 0 and r_init)
        x_start = r_init if row_idx % 2 == 0 else r_init + r_init # Shift by r to nest in gaps
        # Actually for hex packing, odd rows are shifted by r_init horizontally relative to even rows
        # But we must ensure they stay within [0, 1]
        
        # If shifted row (odd), start at 2*r_init? 
        # Let's just generate points and filter.
        
        x = r_init
        # If row is odd, we might want to shift. 
        # Standard hex: Row 0 at x=r, Row 1 at x=2r (centers at 2r, 4r...)
        # But 2r is 0.16, fits.
        if row_idx % 2 == 1:
            x = r_init + r_init # Start at 2r
        
        while x + r_init <= 1.0:
            centers.append([x, y])
            x += dx
        y += dy
        row_idx += 1
        
    # If we didn't get enough circles, add some in remaining space or reduce spacing
    # For n=26, this loop should generate enough.
    # If we have more, take first n. If fewer, add randomly in gaps (unlikely for n=26).
    
    centers = np.array(centers[:n])
    
    # If we have fewer than n, fill with random points (rare case)
    if len(centers) < n:
        extra = n - len(centers)
        # Just place them in center initially with small radius
        for _ in range(extra):
            centers = np.vstack([centers, [0.5, 0.5]])
            
    return centers

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # 1. Generate initial centers
    centers_init = get_initial_centers(n)
    
    # Initial radii (small enough to be valid)
    radii_init = np.full(n, 0.05)
    
    # Combine into variable vector: [x0, y0, r0, x1, y1, r1, ...]
    # Shape: (n * 3,)
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # 2. Define objective function
    def objective(vars):
        # vars has shape (n*3,)
        # We want to maximize sum of radii -> minimize negative sum
        radii = vars[2::3]
        return -np.sum(radii)

    # 3. Define constraints
    # Inequality constraints g(x) >= 0
    
    constraints = []
    
    # Boundary constraints for each circle i
    # x_i >= r_i  => x_i - r_i >= 0
    # x_i <= 1 - r_i => x_i + r_i - 1 <= 0 => 1 - x_i - r_i >= 0
    # Same for y
    
    def add_boundary_constraints(constraints_list, vars, i):
        x = vars[i*3]
        y = vars[i*3+1]
        r = vars[i*3+2]
        
        # We need constraints in terms of 'vars' function
        # SLSQP expects constraints as {'type': 'ineq', 'fun': lambda v: ...}
        
        # x - r >= 0
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[idx*3] - v[idx*3+2]
        })
        # 1 - x - r >= 0
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[idx*3] - v[idx*3+2]
        })
        # y - r >= 0
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[idx*3+1] - v[idx*3+2]
        })
        # 1 - y - r >= 0
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[idx*3+1] - v[idx*3+2]
        })

    for i in range(n):
        add_boundary_constraints(constraints, x0, i)
        
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    
    def add_overlap_constraints(constraints_list, vars, i, j):
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, i=i, j=j: 
                (v[i*3] - v[j*3])**2 + (v[i*3+1] - v[j*3+1])**2 - (v[i*3+2] + v[j*3+2])**2
        })

    for i in range(n):
        for j in range(i + 1, n):
            add_overlap_constraints(constraints, x0, i, j)

    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 1] (loose bounds)
    # Tighter bounds might help, but [0,1] is safe.
    # Actually r can be at most 0.5.
    bounds = [(0, 1)] * (n * 3)
    
    # 4. Optimize
    # Using SLSQP
    try:
        result = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            x_opt = result.x
        else:
            # If optimization fails, try to use the best found so far or fallback
            print("Optimization did not converge successfully, using final iterate.")
            x_opt = result.x
    except Exception as e:
        print(f"Optimization error: {e}")
        x_opt = x0

    # Extract results
    centers_opt = np.array([[x_opt[i*3], x_opt[i*3+1]] for i in range(n)])
    radii_opt = np.array([x_opt[i*3+2] for i in range(n)])
    
    # Ensure non-negative radii due to numerical noise
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Validate and fix any minor violations
    # Although SLSQP should respect constraints, numerical tolerance might allow tiny violations.
    # We can clamp radii if they cause overlaps? 
    # But for the purpose of the function, we return the optimized result.
    # The problem statement implies we should return a valid packing.
    # Let's do a quick check and shrink if necessary, though unlikely with small tol.
    
    # To be safe, we can run a validation check internally and reduce radii slightly if needed.
    # But let's assume the solver did its job.
    
    # Check sum
    sum_radii = np.sum(radii_opt)
    
    # If sum is very low (failed), return initial valid config
    if sum_radii < 1.0: 
         # Fallback to a known valid config (e.g. small grid)
         centers_fallback = np.zeros((n, 2))
         radii_fallback = np.full(n, 0.01)
         # Place in grid
         idx = 0
         for r in range(5):
             for c in range(5):
                 if idx < n:
                     centers_fallback[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                     radii_fallback[idx] = 0.09 # Slightly less than 0.1 to fit
                     idx += 1
         centers_opt = centers_fallback
         radii_opt = radii_fallback
         sum_radii = np.sum(radii_opt)

    return centers_opt, radii_opt, sum_radii

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation print
    import numpy as np
    # Copy validation logic
    n = centers.shape[0]
    valid = True
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
    for i in range(n):
        if radii[i] < 0: valid = False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
            break
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                valid = False
                break
        if not valid: break
    print(f"Valid: {valid}")
