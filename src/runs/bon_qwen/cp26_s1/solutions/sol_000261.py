# sol_000261 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=ed3d5759 sum of radii=2.064586 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with a hexagonal lattice initialization.
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We estimate a reasonable radius to start. 
    # For 26 circles, area ~ 26 * pi * r^2 approx 0.85 (high density).
    # r approx 0.1. We start with a safe radius and positions.
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Hexagonal packing parameters
    # Approximate spacing to fit 26 circles
    # We'll place them in rows. 
    # Row 0: y = r, x = r, 3r, 5r...
    # Row 1: y = r + sqrt(3)r, x = 2r, 4r...
    
    r_init = 0.06 # Safe initial radius
    current_idx = 0
    
    # Generate rows until we have 26 circles
    row = 0
    y_curr = r_init
    
    while current_idx < n:
        # Determine x offset for this row
        # Even rows (0, 2, ...): x starts at r_init, step 2*r_init
        # Odd rows (1, 3, ...): x starts at 2*r_init, step 2*r_init (shifted by r_init relative to even)
        
        x_start = r_init if (row % 2 == 0) else 2 * r_init
        x_step = 2 * r_init
        
        x_curr = x_start
        
        while x_curr + r_init <= 1.0 and current_idx < n:
            centers[current_idx, 0] = x_curr
            centers[current_idx, 1] = y_curr
            radii[current_idx] = r_init
            current_idx += 1
            x_curr += x_step
        
        y_curr += r_init * np.sqrt(3)
        row += 1
    
    # Flatten variables for scipy: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    # 2. Objective Function: Minimize negative sum of radii
    def objective(vars_flat):
        r_sum = 0
        for i in range(n):
            r_sum += vars_flat[3*i + 2]
        return -r_sum

    # 3. Constraints
    # We need:
    # 1. x_i >= r_i  => x_i - r_i >= 0
    # 2. 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # 3. y_i >= r_i
    # 4. 1 - y_i >= r_i
    # 5. dist(i,j)^2 >= (r_i + r_j)^2 => dist^2 - (r_i+r_j)^2 >= 0
    
    cons = []
    
    # Boundary constraints (linear)
    # x_i - r_i >= 0
    for i in range(n):
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.concatenate([np.zeros(3*n), np.ones(1)]).reshape(-1) # Incorrect jac logic below, better to let scipy approx or define properly
            # Actually, defining jacobian manually for all constraints is tedious. 
            # Let's rely on finite differences or simple constraints.
            # But for performance, let's just use fun.
        })
        
    # To make it cleaner and faster, we can define constraint functions that return arrays.
    # But scipy 'ineq' expects a scalar or array. 
    # Let's group them.
    
    # However, defining 100+ individual constraint dicts is verbose.
    # Let's write helper functions.
    
    def boundary_constraints(vars_flat):
        vals = np.zeros(4 * n)
        for i in range(n):
            x = vars_flat[3*i]
            y = vars_flat[3*i+1]
            r = vars_flat[3*i+2]
            idx = i * 4
            vals[idx] = x - r          # Left
            vals[idx+1] = 1 - x - r    # Right
            vals[idx+2] = y - r        # Bottom
            vals[idx+3] = 1 - y - r    # Top
        return vals

    # Overlap constraints: dist^2 - (r1+r2)^2 >= 0
    # This creates ~325 constraints. 
    # We can define a function that returns all of them.
    def overlap_constraints(vars_flat):
        n_cons = 0
        for i in range(n):
            for j in range(i + 1, n):
                n_cons += 1
        
        vals = np.zeros(n_cons)
        k = 0
        for i in range(n):
            xi = vars_flat[3*i]
            yi = vars_flat[3*i+1]
            ri = vars_flat[3*i+2]
            for j in range(i + 1, n):
                xj = vars_flat[3*j]
                yj = vars_flat[3*j+1]
                rj = vars_flat[3*j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum = ri + rj
                vals[k] = dist_sq - rad_sum**2
                k += 1
        return vals

    # Combine constraints into scipy format
    # We can pass a single constraint dict with a function returning an array if method supports it?
    # SLSQP supports array-valued constraint functions returning a vector.
    
    constraint_boundary = {
        'type': 'ineq',
        'fun': boundary_constraints,
        # 'jac': ... # Optional
    }
    
    constraint_overlap = {
        'type': 'ineq',
        'fun': overlap_constraints,
    }
    
    constraints = [constraint_boundary, constraint_overlap]
    
    # 4. Run Optimization
    # Method 'SLSQP' is suitable for bound and non-linear constraints.
    # maxiter might need to be increased for convergence.
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
    
    # 5. Extract Results
    final_vars = res.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_vars[3*i]
        final_centers[i, 1] = final_vars[3*i+1]
        final_radii[i] = final_vars[3*i+2]
        
    # 6. Validation and Correction
    # Ensure strict validity against the validator's tolerance
    # The validator allows 1e-12 error. 
    # We should ensure we are well within or just on the boundary.
    # If optimization pushes slightly outside due to numerical noise, clamp.
    
    # Check boundaries
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        # Ensure inside [0,1] with radius
        # r <= x <= 1-r  => x in [r, 1-r]
        # If x < r, set x = r. If x > 1-r, set x = 1-r.
        # But changing x might cause overlap. 
        # Better to reduce r if x is close to boundary?
        # The constraints were x - r >= 0. If violated slightly, r might be too big for x.
        # Let's enforce r <= x, r <= 1-x, etc. by reducing r.
        
        r_max_x = min(x, 1-x)
        r_max_y = min(y, 1-y)
        r_valid = min(r, r_max_x, r_max_y)
        final_radii[i] = r_valid
        final_radii[i] = max(0.0, final_radii[i]) # Non-negative
        
    # Check overlaps
    # If dist < r1 + r2, we need to reduce radii.
    # A simple iterative fix:
    # Since we optimized for max sum, violations should be minimal.
    # We can just ensure dist >= r1 + r2 - epsilon.
    # If dist < r1 + r2, we scale down radii of both or just one?
    # To maximize sum, we should reduce the sum minimally.
    # But for 26 circles, a global scaling might be safest if errors are small.
    # However, let's just check and fix locally.
    
    for _ in range(10): # Iterate a few times to resolve
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (final_centers[i,0]-final_centers[j,0])**2 + (final_centers[i,1]-final_centers[j,1])**2
                dist = np.sqrt(d2)
                req = final_radii[i] + final_radii[j]
                if dist < req - 1e-12:
                    # Overlap detected. Reduce radii.
                    # To keep them touching, new_r_sum = dist.
                    # Distribute reduction proportionally or equally?
                    # Equally is safer to preserve sum roughly.
                    excess = req - dist
                    reduction = excess / 2.0
                    final_radii[i] -= reduction
                    final_radii[j] -= reduction
                    changed = True
                    # Ensure non-negative
                    final_radii[i] = max(0.0, final_radii[i])
                    final_radii[j] = max(0.0, final_radii[j])
        if not changed:
            break
            
    # Recalculate sum
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
