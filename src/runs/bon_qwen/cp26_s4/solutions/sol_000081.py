# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e5887d00) state=0172c814 sum of radii=2.565293 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

# Global constant for number of circles
N_CIRCLES = 26

def constraint_overlap(vars, i, j):
    """
    Constraint: Distance between centers >= sum of radii.
    Returns value >= 0 if satisfied.
    vars: [x0...xN-1, y0...yN-1, r0...rN-1]
    """
    n = N_CIRCLES
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dr = r[i] + r[j]
    
    return (dx*dx + dy*dy) - (dr*dr)

def constraint_boundary_x(vars, i, side):
    """
    Constraint: Circle inside [0, 1] in x.
    side=0: left (x - r >= 0)
    side=1: right (1 - x - r >= 0)
    """
    n = N_CIRCLES
    x = vars[0:n]
    r = vars[2*n:3*n]
    if side == 0:
        return x[i] - r[i]
    else:
        return 1.0 - x[i] - r[i]

def constraint_boundary_y(vars, i, side):
    """
    Constraint: Circle inside [0, 1] in y.
    side=0: bottom (y - r >= 0)
    side=1: top (1 - y - r >= 0)
    """
    n = N_CIRCLES
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    if side == 0:
        return y[i] - r[i]
    else:
        return 1.0 - y[i] - r[i]

def constraint_radius(vars, i):
    """
    Constraint: Radius >= 0
    """
    n = N_CIRCLES
    r = vars[2*n:3*n]
    return r[i]

def objective(vars):
    """
    Objective: Minimize negative sum of radii (Maximize sum of radii)
    """
    n = N_CIRCLES
    r = vars[2*n:3*n]
    return -np.sum(r)

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = N_CIRCLES
    
    # Initial placement: Hexagonal lattice
    # Start with a safe radius to ensure initial validity
    r_init = 0.04 
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    idx = 0
    row = 0
    
    # Generate hexagonal grid
    while idx < n:
        # y coordinate for current row
        y = r_init + row * np.sqrt(3) * r_init
        
        # If the row is too high, stop
        if y + r_init > 1.0:
            break
            
        # Determine x start coordinate based on row parity for hexagonal packing
        if row % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
        
        col = 0
        while idx < n:
            x = x_start + col * 2 * r_init
            if x + r_init > 1.0:
                break
            
            centers[idx] = [x, y]
            idx += 1
            col += 1
        
        row += 1
    
    # If we didn't fill all circles (unlikely with small r_init), fill the rest
    if idx < n:
        for k in range(idx, n):
            centers[k] = [0.5, 0.5]
            radii[k] = 0.01

    # Concatenate variables: [x1...xN, y1...yN, r1...rN]
    x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    # Define constraints
    cons = []
    
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': constraint_boundary_x, 'args': (i, 0)})
        cons.append({'type': 'ineq', 'fun': constraint_boundary_x, 'args': (i, 1)})
        cons.append({'type': 'ineq', 'fun': constraint_boundary_y, 'args': (i, 0)})
        cons.append({'type': 'ineq', 'fun': constraint_boundary_y, 'args': (i, 1)})
        cons.append({'type': 'ineq', 'fun': constraint_radius, 'args': (i,)})

    # Overlap constraints
    # For N=26, we have 325 pairs. This is computationally feasible.
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': constraint_overlap, 'args': (i, j)})

    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    try:
        # Run optimization
        # SLSQP is suitable for constrained non-linear optimization
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        
        # Extract results
        best_centers = np.column_stack((res.x[0:n], res.x[n:2*n]))
        best_radii = res.x[2*n:3*n]
        
        # Post-processing to ensure validity (handle numerical noise)
        best_radii = np.maximum(best_radii, 0)
        best_centers = np.clip(best_centers, 0, 1)
        
        # Return result
        return best_centers, best_radii, np.sum(best_radii)
    except Exception:
        # Fallback to initial valid packing if optimization fails
        return centers, radii, np.sum(radii)
