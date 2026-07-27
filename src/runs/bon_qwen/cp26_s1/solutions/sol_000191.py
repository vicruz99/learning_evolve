# sol_000191 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9dd6f42d) state=43df69e6 sum of radii=2.607234 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Objective function: minimize negative sum of radii
    def objective(vars):
        r = vars[2 * n:]
        return -np.sum(r)

    # Constraints function
    # Returns an array of constraint values (must be >= 0)
    def constraints_func(vars):
        x = vars[:n]
        y = vars[n:2 * n]
        r = vars[2 * n:]
        
        cons = []
        
        # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
        cons.extend(x - r)
        cons.extend(1 - x - r)
        cons.extend(y - r)
        cons.extend(1 - y - r)
        
        # Non-overlap constraints: dist^2 >= (r1 + r2)^2
        # Using squared distance for smoothness
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                val = dx*dx + dy*dy - dr*dr
                cons.append(val)
                
        return np.array(cons)

    # Generate hexagonal grid initialization
    def get_hex_grid_init(perturb=0.0):
        # Row configuration for 26 circles: 5, 4, 5, 4, 5, 3
        row_counts = [5, 4, 5, 4, 5, 3]
        
        # Parameters for initial placement
        init_r = 0.05
        spacing_x = 2 * init_r
        spacing_y = np.sqrt(3) * init_r
        
        centers_x = []
        centers_y = []
        
        current_y = 0.1 # Starting y position (margin)
        
        for i, count in enumerate(row_counts):
            # Calculate row width to center it in [0, 1]
            row_width = (count - 1) * spacing_x
            start_x = (1.0 - row_width) / 2
            
            # Perturb x position for variety
            row_offset_x = np.random.uniform(-0.02, 0.02) if perturb else 0.0
            
            for k in range(count):
                cx = start_x + k * spacing_x + row_offset_x
                cy = current_y
                
                # Perturb positions slightly
                if perturb:
                    cx += np.random.uniform(-0.02, 0.02)
                    cy += np.random.uniform(-0.02, 0.02)
                
                # Clamp to bounds strictly
                cx = np.clip(cx, init_r + 0.01, 1 - init_r - 0.01)
                cy = np.clip(cy, init_r + 0.01, 1 - init_r - 0.01)
                
                centers_x.append(cx)
                centers_y.append(cy)
            
            current_y += spacing_y
            
        # Construct initial variables vector: [x, y, r]
        vars = np.zeros(3 * n)
        vars[:n] = centers_x
        vars[n:2*n] = centers_y
        vars[2*n:] = init_r
        
        return vars

    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n

    best_result = None
    best_sum_radii = -np.inf

    # Run optimization multiple times with perturbations
    # Using multiple starts to avoid local minima
    for i in range(5):
        try:
            init_vars = get_hex_grid_init(perturb=(i > 0))
            
            res = minimize(
                fun=objective,
                x0=init_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints_func},
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success and res.fun < best_sum_radii: # res.fun is -sum(r), so lower is better
                best_result = res
                best_sum_radii = res.fun
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to a simple grid if optimization fails
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.05)
        # Simple grid fallback
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.17]
                    idx += 1
        return centers, radii, np.sum(radii)

    # Extract results
    x = best_result.x[:n]
    y = best_result.x[n:2 * n]
    r = best_result.x[2 * n:]
    
    centers = np.column_stack((x, y))
    
    # Final validation and cleanup
    # Ensure constraints are strictly satisfied to avoid numerical errors in validation
    # (Though solver should have handled this, tight constraints might drift slightly)
    
    return centers, r, -best_result.fun

# Helper functions must be top level
# (Already defined inside run_packing in thought process, moving out or keeping structure valid)
# The prompt asks for run_packing function. 
# Nested functions are allowed as long as they are not closures capturing mutable state improperly, 
# but standard practice in this context is fine. 
# However, to be safe and follow "Make all helper functions top level" rule strictly:

def objective(vars, n):
    r = vars[2 * n:]
    return -np.sum(r)

def constraints_func(vars, n):
    x = vars[:n]
    y = vars[n:2 * n]
    r = vars[2 * n:]
    
    cons = []
    
    # Boundary constraints
    cons.extend(x - r)
    cons.extend(1 - x - r)
    cons.extend(y - r)
    cons.extend(1 - y - r)
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            val = dx*dx + dy*dy - dr*dr
            cons.append(val)
            
    return np.array(cons)

def get_hex_grid_init(n, perturb=0.0):
    row_counts = [5, 4, 5, 4, 5, 3] # Sums to 26
    
    init_r = 0.05
    spacing_x = 2 * init_r
    spacing_y = np.sqrt(3) * init_r
    
    centers_x = []
    centers_y = []
    
    current_y = 0.1 
    
    for i, count in enumerate(row_counts):
        row_width = (count - 1) * spacing_x
        start_x = (1.0 - row_width) / 2
        row_offset_x = np.random.uniform(-0.02, 0.02) if perturb else 0.0
        
        for k in range(count):
            cx = start_x + k * spacing_x + row_offset_x
            cy = current_y
            
            if perturb:
                cx += np.random.uniform(-0.02, 0.02)
                cy += np.random.uniform(-0.02, 0.02)
            
            cx = np.clip(cx, init_r + 0.01, 1 - init_r - 0.01)
            cy = np.clip(cy, init_r + 0.01, 1 - init_r - 0.01)
            
            centers_x.append(cx)
            centers_y.append(cy)
        
        current_y += spacing_y
        
    vars = np.zeros(3 * n)
    vars[:n] = centers_x
    vars[n:2*n] = centers_y
    vars[2*n:] = init_r
    
    return vars

def run_packing():
    n = 26
    
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n

    best_result = None
    best_sum_radii = -np.inf

    for i in range(5):
        try:
            init_vars = get_hex_grid_init(n, perturb=(i > 0))
            
            # Use lambda to bind n to the functions if necessary, but here we pass n explicitly or use closure
            # Since helper functions take n, we wrap them or just call with args
            # scipy minimize doesn't pass extra args to fun/constraints easily without args=() or lambda
            # We'll use lambda for simplicity
            
            res = minimize(
                fun=lambda v: objective(v, n),
                x0=init_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: constraints_func(v, n)},
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_result = res
                    best_sum_radii = current_sum
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.05)
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < n:
                    centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.17]
                    idx += 1
        return centers, radii, np.sum(radii)

    x = best_result.x[:n]
    y = best_result.x[n:2 * n]
    r = best_result.x[2 * n:]
    
    centers = np.column_stack((x, y))
    
    return centers, r, np.sum(r)
