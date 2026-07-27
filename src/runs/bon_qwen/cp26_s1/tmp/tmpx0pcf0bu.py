import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars_arr):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars_arr.reshape(N_CIRCLES, 3)[:, 2])

def constr_overlap(vars_arr):
    """Non-overlap constraints: d_ij^2 >= (r_i + r_j)^2"""
    cx = vars_arr.reshape(N_CIRCLES, 3)[:, 0]
    cy = vars_arr.reshape(N_CIRCLES, 3)[:, 1]
    r = vars_arr.reshape(N_CIRCLES, 3)[:, 2]
    vals = []
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            vals.append(dx*dx + dy*dy - (r[i] + r[j])**2)
    return np.array(vals)

def constr_boundary(vars_arr):
    """Boundary constraints: circles must stay inside [0,1]x[0,1]"""
    cx = vars_arr.reshape(N_CIRCLES, 3)[:, 0]
    cy = vars_arr.reshape(N_CIRCLES, 3)[:, 1]
    r = vars_arr.reshape(N_CIRCLES, 3)[:, 2]
    vals = []
    for i in range(N_CIRCLES):
        vals.append(cx[i] - r[i])
        vals.append(1.0 - cx[i] - r[i])
        vals.append(cy[i] - r[i])
        vals.append(1.0 - cy[i] - r[i])
    return np.array(vals)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]"""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
    return bounds

def run_packing():
    bounds = get_bounds()
    constraints = [
        {'type': 'ineq', 'fun': constr_overlap},
        {'type': 'ineq', 'fun': constr_boundary}
    ]
    
    best_obj = -np.inf
    best_vars = None
    
    # Run 1: Grid initialization (dense, feasible start)
    cx = np.linspace(0.1, 0.9, 5)
    cy = np.linspace(0.1, 0.9, 5)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    x0 = np.empty(N_CIRCLES * 3)
    for i in range(N_CIRCLES):
        x0[3*i] = cx_grid.flatten()[i]
        x0[3*i+1] = cy_grid.flatten()[i]
        x0[3*i+2] = 0.04  # Feasible starting radius
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-12})
    if -res.fun > best_obj:
        best_obj = -res.fun
        best_vars = res.x.copy()

    # Run 2-8: Random initializations with small radii to explore configurations
    for seed in range(6):
        np.random.seed(seed)
        cx = np.random.rand(N_CIRCLES)
        cy = np.random.rand(N_CIRCLES)
        x0 = np.empty(N_CIRCLES * 3)
        for i in range(N_CIRCLES):
            x0[3*i] = cx[i]
            x0[3*i+1] = cy[i]
            x0[3*i+2] = 0.01  # Small feasible radius to avoid initial infeasibility
            
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-12})
        if -res.fun > best_obj:
            best_obj = -res.fun
            best_vars = res.x.copy()
            
    # Extract results
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    
    return centers, radii, np.sum(radii)