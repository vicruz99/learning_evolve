import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    """Objective: minimize negative sum of radii"""
    return -sum(vars[3*i+2] for i in range(n))

def dist_sq_constraint(vars, i, j):
    """Squared distance constraint: d^2 >= (r_i + r_j)^2"""
    xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
    xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2

def bound_x_min(vars, i):
    return vars[3*i] - vars[3*i+2]
def bound_x_max(vars, i):
    return 1.0 - vars[3*i] - vars[3*i+2]
def bound_y_min(vars, i):
    return vars[3*i+1] - vars[3*i+2]
def bound_y_max(vars, i):
    return 1.0 - vars[3*i+1] - vars[3*i+2]

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initial hexagonal-like grid: rows with 6, 5, 6, 5, 4 circles
    rows_counts = [6, 5, 6, 5, 4]
    r_init = 0.05
    y_vals = np.linspace(r_init + 0.08, 1 - r_init - 0.08, len(rows_counts))
    
    centers = []
    for i, count in enumerate(rows_counts):
        y = y_vals[i]
        if count == 0:
            continue
        if count == 1:
            xs = [0.5]
        else:
            total_width = (count - 1) * 2 * r_init
            x_start = (1.0 - total_width) / 2.0
            xs = np.linspace(x_start + r_init, x_start + total_width + r_init, count)
            
        for x in xs:
            centers.append([x + np.random.uniform(-0.002, 0.002), 
                            y + np.random.uniform(-0.002, 0.002)])
            
    centers = np.array(centers)
    radii0 = np.full(n, r_init)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii0[i]
        
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(1e-6, 0.5) for _ in range(n)]
    
    constraints = []
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': dist_sq_constraint, 'args': (i, j)})
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': bound_x_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_x_max, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_y_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_y_max, 'args': (i,)})
        
    res = minimize(compute_objective, x0, method='SLSQP', args=(n,), bounds=bounds, 
                   constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False})
                   
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [res.x[3*i], res.x[3*i+1]]
        final_radii[i] = res.x[3*i+2]
        
    # Strict boundary enforcement to handle numerical tolerances
    final_radii = np.clip(final_radii, 1e-9, None)
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, float(np.sum(final_radii))