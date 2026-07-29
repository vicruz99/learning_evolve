# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=7bd361d5 sum of radii=2.409887 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars, n):
    """
    Compute inequality constraints for the optimization problem.
    Returns an array of constraint values that must be >= 0.
    """
    # 4 constraints per circle for boundaries
    # n*(n-1)/2 constraints for pairwise non-overlap
    c = np.empty(4*n + n*(n-1)//2)
    idx = 0
    
    # Boundary constraints
    for i in range(n):
        c[idx] = vars[3*i] - vars[3*i+2]          # x_i - r_i >= 0
        idx += 1
        c[idx] = 1.0 - vars[3*i] - vars[3*i+2]    # 1 - x_i - r_i >= 0
        idx += 1
        c[idx] = vars[3*i+1] - vars[3*i+2]        # y_i - r_i >= 0
        idx += 1
        c[idx] = 1.0 - vars[3*i+1] - vars[3*i+2]  # 1 - y_i - r_i >= 0
        idx += 1
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = vars[3*i] - vars[3*j]
            dy = vars[3*i+1] - vars[3*j+1]
            dr = vars[3*i+2] + vars[3*j+2]
            c[idx] = dx*dx + dy*dy - dr*dr        # dist^2 - (r_i+r_j)^2 >= 0
            idx += 1
    return c

def objective(vars, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def generate_layout(n):
    """Generate an initial hexagonal layout for n circles."""
    rows = [6, 5, 6, 5, 4]
    centers = np.zeros((n, 2))
    s = 0.18 # Horizontal spacing parameter
    h = s * np.sqrt(3) / 2 # Vertical spacing for hex packing
    
    y = 0.3
    idx = 0
    for r_idx, count in enumerate(rows):
        x_start = (1.0 - (count - 1) * s) / 2
        for k in range(count):
            x = x_start + k * s
            if r_idx % 2 == 1:
                x += s / 2 # Offset for hexagonal staggering
            centers[idx] = [x, y]
            idx += 1
        y += h
        
    # Center the entire layout vertically within the unit square
    y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
    centers[:, 1] += 0.5 - (y_min + y_max) / 2
    
    radii = np.full(n, 0.08)
    return centers, radii

def run_packing():
    """Run the packing optimization and return valid centers, radii, and sum."""
    n = 26
    centers, radii = generate_layout(n)
    
    # Flatten to optimization variable vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for x, y in [0,1] and r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Setup constraints and optimization
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    res = minimize(
        objective, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 5000, 'ftol': 1e-12}
    )
    
    x_opt = res.x
    centers_opt = np.array([[x_opt[3*i], x_opt[3*i+1]] for i in range(n)])
    radii_opt = x_opt[2::3]
    
    # Ensure non-negative radii and apply small safety margin for strict validity
    radii_opt = np.maximum(radii_opt, 0.0)
    radii_opt -= 1e-6
    
    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii
