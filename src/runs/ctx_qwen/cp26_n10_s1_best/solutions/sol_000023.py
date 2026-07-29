# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 764eb384) state=1725ae61 sum of radii=2.567820 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

# Global constants
N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    # Radii are at indices 2, 5, 8, ... (index 2 + 3*i)
    radii = vars[2::3]
    return -np.sum(radii)

def boundary_constraint_x_min(vars, i):
    """Constraint: x_i >= r_i  =>  x_i - r_i >= 0"""
    return vars[i*3] - vars[i*3+2]

def boundary_constraint_x_max(vars, i):
    """Constraint: x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0"""
    return 1.0 - vars[i*3] - vars[i*3+2]

def boundary_constraint_y_min(vars, i):
    """Constraint: y_i >= r_i  =>  y_i - r_i >= 0"""
    return vars[i*3+1] - vars[i*3+2]

def boundary_constraint_y_max(vars, i):
    """Constraint: y_i <= 1 - r_i  =>  1 - y_i - r_i >= 0"""
    return 1.0 - vars[i*3+1] - vars[i*3+2]

def overlap_constraint(vars, i, j):
    """Constraint: dist(i,j) >= r_i + r_j  =>  dist^2 - (r_i+r_j)^2 >= 0"""
    dx = vars[i*3] - vars[j*3]
    dy = vars[i*3+1] - vars[j*3+1]
    dr = vars[i*3+2] + vars[j*3+2]
    return dx*dx + dy*dy - dr*dr

def run_packing():
    n = N_CIRCLES
    
    # Generate initial feasible configuration
    # A dense grid of points to start with
    np.random.seed(42)
    # 6 columns, 5 rows = 30 points. We need 26.
    # This creates a compact block of circles.
    x_coords = np.linspace(0.1, 0.9, 6) 
    y_coords = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Take first 26 points
    centers_init = grid_points[:n].copy()
    # Add small random noise to break symmetry
    centers_init += np.random.normal(0, 0.01, centers_init.shape)
    # Ensure within bounds [0.05, 0.95] for safety
    centers_init = np.clip(centers_init, 0.05, 0.95)
    
    # Initial radii small enough to ensure no overlap initially
    # Grid spacing is approx 0.16, so r=0.04 is very safe (diam=0.08)
    radii_init = np.full(n, 0.04)
    
    # Prepare initial variable vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[i*3] = centers_init[i, 0]
        x0[i*3+1] = centers_init[i, 1]
        x0[i*3+2] = radii_init[i]
        
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Build constraints list
    cons = []
    
    # Boundary constraints for each circle
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': boundary_constraint_x_min, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_constraint_x_max, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_constraint_y_min, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_constraint_y_max, 'args': (i,)})
        
    # Overlap constraints for each pair
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})
            
    # Run optimization
    try:
        res = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        if res.success:
            best_centers = np.array([[res.x[i*3], res.x[i*3+1]] for i in range(n)])
            best_radii = res.x[2::3]
            best_sum = float(-res.fun)
        else:
            # Fallback if optimization fails
            # Use a safe radius for the initial centers
            best_centers = centers_init
            best_radii = np.full(n, 0.04)
            best_sum = float(np.sum(best_radii))
    except Exception:
        best_centers = centers_init
        best_radii = np.full(n, 0.04)
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, best_sum
