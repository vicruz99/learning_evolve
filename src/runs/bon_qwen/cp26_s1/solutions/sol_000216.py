# sol_000216 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=769877dd sum of radii=2.485862 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def bound_x_ge_r(x, i):
    """Constraint: x_i >= r_i"""
    return x[3*i] - x[3*i+2]

def bound_x_le_1_r(x, i):
    """Constraint: 1 - x_i >= r_i"""
    return 1.0 - x[3*i] - x[3*i+2]

def bound_y_ge_r(x, i):
    """Constraint: y_i >= r_i"""
    return x[3*i+1] - x[3*i+2]

def bound_y_le_1_r(x, i):
    """Constraint: 1 - y_i >= r_i"""
    return 1.0 - x[3*i+1] - x[3*i+2]

def overlap_constraint(x, i, j):
    """Constraint: dist^2 >= (r_i + r_j)^2"""
    dx = x[3*i] - x[3*j]
    dy = x[3*i+1] - x[3*j+1]
    dr = x[3*i+2] + x[3*j+2]
    return dx**2 + dy**2 - dr**2

def obj_fun(x):
    """Objective: minimize negative sum of radii"""
    return -np.sum(x[2::3])

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initialize centers on a hexagonal lattice
    centers = []
    y = 0.12
    dy = 0.18
    rows = [6, 6, 6, 6, 2]
    row_idx = 0
    for count in rows:
        if len(centers) + count > n:
            count = n - len(centers)
        dx = 0.20
        x_start = (1.0 - (count - 1) * dx) / 2.0
        if row_idx % 2 == 1:
            x_start += dx / 2.0
        for _ in range(count):
            centers.append([x_start, y])
            x_start += dx
        y += dy
        row_idx += 1
        
    centers = np.array(centers)
    # Small perturbation to break symmetry and aid exploration
    centers += np.random.randn(n, 2) * 0.005
    
    # Flatten to optimization vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = 0.08  # Initial feasible radius
        
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Build constraints list
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': bound_x_ge_r, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_x_le_1_r, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_y_ge_r, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_y_le_1_r, 'args': (i,)})
        
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})
            
    # Run optimization
    res = minimize(obj_fun, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
                   
    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = res.x[3*i:3*i+2]
        final_radii[i] = res.x[3*i+2]
        
    return final_centers, final_radii, float(np.sum(final_radii))
