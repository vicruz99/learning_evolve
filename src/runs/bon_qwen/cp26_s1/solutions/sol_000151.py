# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e234a3e4) state=a958b5e8 sum of radii=2.588818 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def get_constraints(centers, radii):
    """
    Helper to check constraints and return violation magnitudes.
    Returns an array of constraint values.
    For equality constraints, value should be 0.
    For inequality constraints (g(x) >= 0), value should be positive.
    Here we define constraints as g(x) >= 0.
    """
    n = centers.shape[0]
    constraints = []
    
    # Boundary constraints
    # x - r >= 0
    for i in range(n):
        constraints.append(centers[i, 0] - radii[i])
    # 1 - (x + r) >= 0  =>  x + r <= 1
    for i in range(n):
        constraints.append(1.0 - (centers[i, 0] + radii[i]))
    # y - r >= 0
    for i in range(n):
        constraints.append(centers[i, 1] - radii[i])
    # 1 - (y + r) >= 0
    for i in range(n):
        constraints.append(1.0 - (centers[i, 1] + radii[i]))
    
    # Radius non-negativity
    for i in range(n):
        constraints.append(radii[i])
        
    # Non-overlap constraints
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            constraints.append(dist_sq - r_sum*r_sum)
            
    return np.array(constraints)

def objective(params, n):
    """
    Objective function to minimize: -sum(radii)
    params: array of shape (n * 3)
    """
    radii = params[2::3]
    return -np.sum(radii)

def run_packing():
    n = 26
    
    # --- Initialization ---
    # Create a hexagonal grid of points
    # Spacing of 1.0 in a theoretical coordinate system
    # We will generate more points than needed and pick the best ones,
    # or just generate a specific pattern.
    
    # Let's try to pack them in a rectangular region of a hex grid
    # 5 rows, roughly 5-6 columns
    points = []
    row_height = math.sqrt(3) / 2 # vertical distance between rows for horizontal spacing 1.0
    
    # We'll generate a grid and scale it later
    # Let's aim for a grid that fits in [0, 1] x [0, 1] roughly
    # But first, let's just generate points in a large grid and select
    
    # Alternative: Place points in a dense cluster
    # 6 columns, 5 rows is 30. Let's do 5 rows.
    # Row 0: 6 points
    # Row 1: 5 points (shifted)
    # Row 2: 6 points
    # Row 3: 5 points
    # Row 4: 4 points (to make 26? 6+5+6+5+4 = 26)
    
    # Let's define coordinates for 26 points in a hexagonal pattern
    # Spacing 2.0 (diameter 2, radius 1)
    cols_counts = [6, 5, 6, 5, 4]
    initial_points = []
    
    y_curr = 0.0
    for r_idx, count in enumerate(cols_counts):
        # Shift odd rows by 1.0 (half spacing)
        x_offset = 1.0 if r_idx % 2 == 1 else 0.0
        for c_idx in range(count):
            x = x_offset + 2.0 * c_idx
            y = y_curr
            initial_points.append([x, y])
        y_curr += math.sqrt(3) # Vertical spacing for touching circles (radius 1)
        
    initial_points = np.array(initial_points)
    
    # Scale and center to fit in [0, 1] x [0, 1]
    # Current bounds
    min_x, min_y = np.min(initial_points, axis=0)
    max_x, max_y = np.max(initial_points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # We want to fit this into [0, 1] with some margin
    # Let's scale by factor s such that width*s <= 1 and height*s <= 1
    # But we also need to leave space for radius.
    # Actually, let's just center it and scale to fill, then optimization will shrink radii if needed.
    # But wait, if we scale to fill exactly, radii might be 0 or negative in constraint sense if we don't account for them.
    # The points are centers.
    
    # Let's scale the coordinates to be within [0.1, 0.9] roughly, so radius can be around 0.05-0.1
    scale_x = 0.8 / width if width > 0 else 1.0
    scale_y = 0.8 / height if height > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    centers_init = initial_points * scale
    # Center in unit square
    centers_init -= np.min(centers_init, axis=0)
    centers_init += (1.0 - np.max(centers_init, axis=0)) / 2.0
    
    # Initial radii: small value, e.g., 0.05
    radii_init = np.full(n, 0.05)
    
    # Assemble initial params
    # Format: x0, y0, r0, x1, y1, r1, ...
    params_init = np.zeros(n * 3)
    params_init[0::3] = centers_init[:, 0]
    params_init[1::3] = centers_init[:, 1]
    params_init[2::3] = radii_init
    
    # --- Optimization ---
    # We will run multiple trials with slight perturbations to avoid local minima
    best_params = params_init.copy()
    best_val = objective(params_init, n)
    
    # Check feasibility of initial guess
    # If not feasible, constraints might fail. 
    # The optimizer handles inequality constraints g(x) >= 0.
    # We need to make sure initial guess is feasible or at least close.
    # With r=0.05 and spread out centers, it should be feasible.
    
    # Define constraints for scipy
    # Bounds: x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r (upper bound 1 is safe)
        
    # Constraints callback
    def constr_func(params):
        c = np.zeros(n * 3) # Dummy shape, will overwrite
        # We need to return a list of constraint dictionaries or a single dict
        # Actually scipy.minimize accepts a list of constraint dicts.
        pass 
        
    # Construct constraint list
    constraints_list = []
    
    # 1. Boundary constraints
    for i in range(n):
        # x - r >= 0
        def make_con_x_lower(i):
            def con(params):
                return params[3*i] - params[3*i + 2]
            return con
        constraints_list.append({'type': 'ineq', 'fun': make_con_x_lower(i)})
        
        # 1 - (x + r) >= 0
        def make_con_x_upper(i):
            def con(params):
                return 1.0 - (params[3*i] + params[3*i + 2])
            return con
        constraints_list.append({'type': 'ineq', 'fun': make_con_x_upper(i)})
        
        # y - r >= 0
        def make_con_y_lower(i):
            def con(params):
                return params[3*i + 1] - params[3*i + 2]
            return con
        constraints_list.append({'type': 'ineq', 'fun': make_con_y_lower(i)})
        
        # 1 - (y + r) >= 0
        def make_con_y_upper(i):
            def con(params):
                return 1.0 - (params[3*i + 1] + params[3*i + 2])
            return con
        constraints_list.append({'type': 'ineq', 'fun': make_con_y_upper(i)})

    # 2. Overlap constraints
    # This creates many constraints. For n=26, 325 constraints.
    # Might be slow to pass as list of dicts?
    # It's fine.
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_con(i, j):
                def con(params):
                    xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
                    xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    r_sum_sq = (ri + rj)**2
                    return dist_sq - r_sum_sq
                return con
            constraints_list.append({'type': 'ineq', 'fun': make_overlap_con(i, j)})

    # Run optimization
    # We try a few random restarts
    rng = np.random.default_rng(42)
    
    for trial in range(5):
        # Perturb initial guess slightly
        current_params = params_init.copy()
        if trial > 0:
            noise = rng.uniform(-0.01, 0.01, size=current_params.shape)
            # Clamp bounds for noise
            noise[0::3] = np.clip(noise[0::3], -0.1, 0.1) # x
            noise[1::3] = np.clip(noise[1::3], -0.1, 0.1) # y
            noise[2::3] = np.abs(noise[2::3]) * 0.01 # r positive noise
            current_params += noise
            # Ensure bounds
            current_params[0::3] = np.clip(current_params[0::3], 0, 1)
            current_params[1::3] = np.clip(current_params[1::3], 0, 1)
            current_params[2::3] = np.clip(current_params[2::3], 0, 1)
            
        try:
            res = opt.minimize(
                objective,
                current_params,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            if res.success or res.nit > 100: # Check if it ran
                val = -res.fun # Objective was negative sum
                if val > best_val:
                    # Verify constraints roughly
                    params = res.x
                    centers = np.column_stack((params[0::3], params[1::3]))
                    radii = params[2::3]
                    # Quick check
                    valid = True
                    for i in range(n):
                        if centers[i,0] - radii[i] < -1e-6 or centers[i,0] + radii[i] > 1 + 1e-6:
                            valid = False; break
                        if centers[i,1] - radii[i] < -1e-6 or centers[i,1] + radii[i] > 1 + 1e-6:
                            valid = False; break
                    if valid:
                        best_val = val
                        best_params = params.copy()
        except Exception as e:
            print(f"Trial {trial} failed: {e}")
            continue

    # Extract results
    centers = np.column_stack((best_params[0::3], best_params[1::3]))
    radii = best_params[2::3]
    sum_radii = np.sum(radii)
    
    # Final validation and cleanup
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Fallback to a safe grid packing if optimization failed
        # Simple grid
        centers_safe = np.zeros((n, 2))
        radii_safe = np.zeros(n)
        r_safe = 0.09
        count = 0
        for row in range(5):
            for col in range(6):
                if count < n:
                    x = 0.1 + col * 0.18
                    y = 0.1 + row * 0.18
                    centers_safe[count] = [x, y]
                    radii_safe[count] = r_safe
                    count += 1
        centers = centers_safe
        radii = radii_safe
        sum_radii = np.sum(radii)

    return centers, radii, float(sum_radii)
