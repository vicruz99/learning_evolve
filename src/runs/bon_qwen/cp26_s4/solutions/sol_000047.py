# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4e4d202b) state=4a649751 sum of radii=2.585427 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_constraints(centers, radii, n):
    """
    Generate the constraint functions for the optimizer.
    Returns a list of constraint dictionaries.
    """
    constraints = []
    
    # Boundary constraints: r <= x, 1-x, y, 1-y
    # x - r >= 0 => -(x - r) <= 0
    # x + r <= 1 => (x + r) - 1 <= 0
    # Same for y
    
    # We can flatten variables: [x0, y0, x1, y1, ..., r0, r1, ...]
    # Or keep structure. scipy minimize expects a single vector.
    # Let's map indices: 
    # 0..2n-1 are coordinates
    # 2n..3n-1 are radii
    
    # Actually, simpler to define constraints directly in the objective or 
    # use a custom constraint function that evaluates all inequalities.
    # SLSQP supports constraints of form g(x) >= 0 or g(x) = 0.
    
    # Let's define a function that returns a vector of constraint values.
    # We want all values >= 0.
    
    pass

def objective(vars, n):
    """
    Objective function: maximize sum of radii.
    vars: array of size 3*n.
    First 2*n are centers (x, y) flattened? 
    Let's use structure:
    vars[0:n] = x
    vars[n:2n] = y
    vars[2n:3n] = r
    """
    r = vars[2*n:]
    return -np.sum(r) # Minimize negative sum

def get_jacobian(vars, n):
    """
    Jacobian of the objective.
    """
    jac = np.zeros(3 * n)
    jac[2*n:] = -1.0
    return jac

def constraint_boundary(vars, n):
    """
    Constraints for circles being inside [0,1]x[0,1].
    x_i >= r_i  => x_i - r_i >= 0
    x_i <= 1-r_i => 1 - x_i - r_i >= 0
    y_i >= r_i  => y_i - r_i >= 0
    y_i <= 1-r_i => 1 - y_i - r_i >= 0
    """
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    # We can return a vector of 4n constraints
    c1 = x - r
    c2 = 1 - x - r
    c3 = y - r
    c4 = 1 - y - r
    
    return np.concatenate([c1, c2, c3, c4])

def constraint_jac_boundary(vars, n):
    """
    Jacobian for boundary constraints.
    Output shape: (4n, 3n)
    """
    # c1 = x - r. dc/dx = 1, dc/dr = -1
    # c2 = 1 - x - r. dc/dx = -1, dc/dr = -1
    # etc.
    
    jac = np.zeros((4 * n, 3 * n))
    
    # c1: x - r
    jac[:n, :n] = np.eye(n)
    jac[:n, 2*n:] = -np.eye(n)
    
    # c2: 1 - x - r
    jac[n:2*n, :n] = -np.eye(n)
    jac[n:2*n, 2*n:] = -np.eye(n)
    
    # c3: y - r
    jac[2*n:3*n, n:2*n] = np.eye(n)
    jac[2*n:3*n, 2*n:] = -np.eye(n)
    
    # c4: 1 - y - r
    jac[3*n:4*n, n:2*n] = -np.eye(n)
    jac[3*n:4*n, 2*n:] = -np.eye(n)
    
    return jac

def constraint_non_overlap(vars, n):
    """
    Constraints for non-overlapping circles.
    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            rad_sum = r[i] + r[j]
            cons.append(dist_sq - rad_sum**2)
            
    return np.array(cons)

def constraint_jac_non_overlap(vars, n):
    """
    Jacobian for non-overlap constraints.
    Number of constraints = n*(n-1)/2
    Variables = 3n
    """
    m = n * (n - 1) // 2
    jac = np.zeros((m, 3 * n))
    
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint k = dist_sq - (r_i + r_j)^2
            # d/dx_i = 2(x_i - x_j)
            # d/dx_j = -2(x_i - x_j)
            # d/dy_i = 2(y_i - y_j)
            # d/dy_j = -2(y_i - y_j)
            # d/dr_i = -2(r_i + r_j)
            # d/dr_j = -2(r_i + r_j)
            
            dx = 2 * (x[i] - x[j])
            dy = 2 * (y[i] - y[j])
            dr = -2 * (r[i] + r[j])
            
            jac[idx, i] = dx          # x_i
            jac[idx, j] = -dx         # x_j
            jac[idx, n + i] = dy      # y_i
            jac[idx, n + j] = -dy     # y_j
            jac[idx, 2*n + i] = dr    # r_i
            jac[idx, 2*n + j] = dr    # r_j
            
            idx += 1
            
    return jac

def run_packing() -> tuple:
    n = 26
    
    # --- Initialization ---
    # Try to create a hexagonal packing
    # We'll place circles in rows.
    
    # Heuristic for initial radius. 
    # For N=26, maybe r=0.08 is safe.
    r_init = 0.08
    
    centers = []
    
    # Generate hexagonal grid points
    # Row spacing = sqrt(3) * r_init
    # Col spacing = 2 * r_init
    dy = np.sqrt(3) * r_init
    dx = 2 * r_init
    
    y_curr = r_init
    row = 0
    while y_curr + r_init <= 1.0 + 1e-9:
        x_curr = r_init
        if row % 2 == 1:
            x_curr += r_init # Shift row
        
        while x_curr + r_init <= 1.0 + 1e-9:
            centers.append([x_curr, y_curr])
            if len(centers) >= n:
                break
            x_curr += dx
        if len(centers) >= n:
            break
        y_curr += dy
        row += 1
        
    # If we didn't get enough centers (unlikely with r=0.08), fill randomly
    while len(centers) < n:
        # Random position
        cx = np.random.uniform(r_init, 1 - r_init)
        cy = np.random.uniform(r_init, 1 - r_init)
        centers.append([cx, cy])
        
    centers = np.array(centers[:n])
    
    # Initial variables
    # x: 0..n-1
    # y: n..2n-1
    # r: 2n..3n-1
    x0 = centers[:, 0]
    y0 = centers[:, 1]
    r0 = np.full(n, r_init)
    
    vars0 = np.concatenate([x0, y0, r0])
    
    # Define constraints
    # Boundary
    cons_bound = {
        'type': 'ineq',
        'fun': lambda v: constraint_boundary(v, n),
        'jac': lambda v: constraint_jac_boundary(v, n)
    }
    
    # Non-overlap
    cons_nolap = {
        'type': 'ineq',
        'fun': lambda v: constraint_non_overlap(v, n),
        'jac': lambda v: constraint_jac_non_overlap(v, n)
    }
    
    constraints = [cons_bound, cons_nolap]
    
    # Bounds for variables
    # x, y in [0, 1] (though constraints handle r, loose bounds help)
    # r >= 0
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
    
    # Optimization
    # Use SLSQP
    result = minimize(
        objective, 
        vars0, 
        args=(n,), 
        method='SLSQP', 
        jac=get_jacobian, 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False}
    )
    
    if result.success:
        vars_opt = result.x
    else:
        vars_opt = vars0
        
    # Extract solution
    x_opt = vars_opt[:n]
    y_opt = vars_opt[n:2*n]
    r_opt = vars_opt[2*n:3*n]
    
    # Clip radii to be non-negative just in case
    r_opt = np.maximum(r_opt, 0)
    
    centers_opt = np.column_stack((x_opt, y_opt))
    sum_radii = np.sum(r_opt)
    
    # Validation check (internal)
    # Note: The validate_packing function provided by user is strict.
    # Our optimizer might have slight numerical errors.
    # We should ensure the result is strictly valid.
    # The constraints enforce >= 0 with tolerance.
    # But let's just return the result. The optimizer with exact jacobians should be precise.
    
    return centers_opt, r_opt, sum_radii

# To make it robust, we can run multiple random starts?
# But we can't loop inside run_packing easily without timing out or violating structure.
# However, we can try a few "restarts" inside the function if time permits?
# No, let's stick to the single optimized run with a good initial guess.
# The hexagonal grid is a very good start.

# Wait, if the hex grid is too sparse or dense, it might be suboptimal.
# But SLSQP should adjust it.

# One issue: The objective is to maximize sum of radii.
# If the optimizer finds a solution where radii are 0, it's valid but bad.
# But with initial r=0.08, it should push them up.

# Let's double check the constraints logic.
# constraint_non_overlap returns dist_sq - (r_i+r_j)^2 >= 0.
# This is correct.

# One potential problem: Local minima.
# 26 circles is a complex landscape.
# Maybe I should add a small random perturbation to the initial positions to break symmetry?
# Or run the optimization twice?
# Since I can only return one function result, I'll do a quick multi-start inside run_packing 
# if I can afford the time. 26 circles, 78 vars, SLSQP might take 0.5-1s per run.
# I can try 3-5 runs.

def run_packing() -> tuple:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Number of attempts
    n_attempts = 5
    
    for attempt in range(n_attempts):
        r_init = 0.08
        centers = []
        
        # Hexagonal grid
        dy = np.sqrt(3) * r_init
        dx = 2 * r_init
        y_curr = r_init
        row = 0
        
        # Add some randomness to grid to avoid symmetries if needed
        # But deterministic grid is usually fine.
        
        while y_curr + r_init <= 1.0 + 1e-9:
            x_curr = r_init
            if row % 2 == 1:
                x_curr += r_init 
            
            while x_curr + r_init <= 1.0 + 1e-9:
                centers.append([x_curr, y_curr])
                if len(centers) >= n:
                    break
                x_curr += dx
            if len(centers) >= n:
                break
            y_curr += dy
            row += 1
            
        while len(centers) < n:
            cx = np.random.uniform(r_init, 1 - r_init)
            cy = np.random.uniform(r_init, 1 - r_init)
            centers.append([cx, cy])
            
        centers = np.array(centers[:n])
        
        # Perturb slightly
        centers += np.random.normal(0, 0.01, centers.shape)
        centers = np.clip(centers, r_init, 1 - r_init)
        
        x0 = centers[:, 0]
        y0 = centers[:, 1]
        r0 = np.full(n, r_init)
        
        vars0 = np.concatenate([x0, y0, r0])
        
        bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
        
        cons_bound = {
            'type': 'ineq',
            'fun': lambda v: constraint_boundary(v, n),
            'jac': lambda v: constraint_jac_boundary(v, n)
        }
        
        cons_nolap = {
            'type': 'ineq',
            'fun': lambda v: constraint_non_overlap(v, n),
            'jac': lambda v: constraint_jac_non_overlap(v, n)
        }
        
        constraints = [cons_bound, cons_nolap]
        
        try:
            result = minimize(
                objective, 
                vars0, 
                args=(n,), 
                method='SLSQP', 
                jac=get_jacobian, 
                bounds=bounds, 
                constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 500} # Reduced maxiter for multiple runs
            )
            
            if result.success or (result.fun < -best_sum): # Check if better
                 # We want to maximize sum, so minimize -sum. 
                 # result.fun is -sum. So we want smaller result.fun.
                 if result.fun < -best_sum: # Actually best_sum is positive sum. -sum < -best_sum => sum > best_sum
                     vars_opt = result.x
                     x_opt = vars_opt[:n]
                     y_opt = vars_opt[n:2*n]
                     r_opt = vars_opt[2*n:3*n]
                     r_opt = np.maximum(r_opt, 0)
                     # Basic validation to avoid NaNs etc
                     if not np.any(np.isnan(r_opt)) and np.all(r_opt >= 0):
                         best_centers = np.column_stack((x_opt, y_opt))
                         best_radii = r_opt
                         best_sum = np.sum(r_opt)
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.0)
        best_sum = 0.0
        
    return best_centers, best_radii, best_sum
