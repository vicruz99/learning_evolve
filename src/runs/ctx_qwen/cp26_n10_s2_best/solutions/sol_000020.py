# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b9ac6cc) state=fea4b3d4 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def make_boundary_xr_constraint(i):
    def fun(v):
        return v[3*i] - v[3*i+2]
    return fun

def make_boundary_1xr_constraint(i):
    def fun(v):
        return 1.0 - v[3*i] - v[3*i+2]
    return fun

def make_boundary_yr_constraint(i):
    def fun(v):
        return v[3*i+1] - v[3*i+2]
    return fun

def make_boundary_1yr_constraint(i):
    def fun(v):
        return 1.0 - v[3*i+1] - v[3*i+2]
    return fun

def make_overlap_constraint(i, j):
    def fun(v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        dist = np.sqrt(dx*dx + dy*dy)
        return dist - (v[3*i+2] + v[3*j+2])
    return fun

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal packing
    # Start with a radius that fits 26 circles easily to ensure valid start
    r_init = 0.09
    centers_init = []
    y = r_init
    row_idx = 0
    
    while len(centers_init) < n:
        # Hexagonal lattice: alternate row offsets
        if row_idx % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
        
        x = x_start
        # Generate points in current row
        while x <= 1 - r_init:
            centers_init.append([x, y])
            x += 2 * r_init
        
        y += r_init * np.sqrt(3)
        row_idx += 1
        
    centers_init = centers_init[:n]
    centers_init = np.array(centers_init)
    
    # 2. Prepare Optimization Variables
    # Vector structure: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(vars):
        return -np.sum(vars[2::3])
        
    constraints_list = []
    
    # Add boundary constraints for each circle
    for i in range(n):
        constraints_list.append({'type': 'ineq', 'fun': make_boundary_xr_constraint(i)})
        constraints_list.append({'type': 'ineq', 'fun': make_boundary_1xr_constraint(i)})
        constraints_list.append({'type': 'ineq', 'fun': make_boundary_yr_constraint(i)})
        constraints_list.append({'type': 'ineq', 'fun': make_boundary_1yr_constraint(i)})
        
    # Add non-overlap constraints for each pair
    for i in range(n):
        for j in range(i + 1, n):
            constraints_list.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
            
    # 3. Optimize
    try:
        # SLSQP is suitable for non-linear constrained optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
        final_vars = res.x
    except Exception:
        # Fallback to initial configuration if optimization fails
        final_vars = x0
        
    # 4. Extract Results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_vars[3*i]
        final_centers[i, 1] = final_vars[3*i+1]
        final_radii[i] = final_vars[3*i+2]
        
    # Ensure radii are non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Check for validity (NaNs)
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        # Return a trivial valid packing if optimization produced garbage
        # e.g. 26 tiny circles
        final_centers = np.array([[0.1, 0.1] for _ in range(n)])
        final_radii = np.zeros(n)
        
    return final_centers, final_radii, np.sum(final_radii)
