import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: maximize sum of radii."""
    return -np.sum(vars[2::3])

def con_fun(vars):
    """Constraint function: ensures circles are inside square and don't overlap."""
    n = N_CIRCLES
    c = vars.reshape((n, 3))
    vals = []
    
    # Boundary constraints: circles must be within [0,1]^2
    # x - r >= 0
    vals.extend(c[:, 0] - c[:, 2])
    # 1 - x - r >= 0
    vals.extend(1.0 - c[:, 0] - c[:, 2])
    # y - r >= 0
    vals.extend(c[:, 1] - c[:, 2])
    # 1 - y - r >= 0
    vals.extend(1.0 - c[:, 1] - c[:, 2])
    
    # Pairwise non-overlap constraints: dist^2 >= (r1 + r2)^2
    c_xy = c[:, :2]
    c_r = c[:, 2]
    
    dx = c_xy[:, 0, np.newaxis] - c_xy[np.newaxis, :, 0]
    dy = c_xy[:, 1, np.newaxis] - c_xy[np.newaxis, :, 1]
    dist_sq = dx**2 + dy**2
    
    upper_tri = np.triu_indices(n, k=1)
    dist_sq_upper = dist_sq[upper_tri]
    
    sum_r = c_r[:, np.newaxis] + c_r[np.newaxis, :]
    sum_r_upper = sum_r[upper_tri]
    
    vals.extend(dist_sq_upper - sum_r_upper**2)
    
    return np.array(vals)

def run_packing():
    n = N_CIRCLES
    
    # Initial positions: hexagonal pattern
    centers_init = np.zeros((n, 2))
    idx = 0
    
    s = 0.16
    y = 0.08
    rows = [5, 4, 5, 4, 5, 3]  # 26 circles total
    
    for i, ncols in enumerate(rows):
        x_start = 0.08 if i % 2 == 0 else 0.08 + s/2
        for j in range(ncols):
            centers_init[idx] = [x_start + j * s, y]
            idx += 1
        y += s * np.sqrt(3)/2
        
    radii_init = np.full(n, 0.05)
    
    # Flatten initial guess
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    constr = {'type': 'ineq', 'fun': con_fun}
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constr, 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
        
        c_opt = res.x.reshape((n, 3))
        centers = c_opt[:, :2].copy()
        radii = c_opt[:, 2].copy()
    except Exception:
        # Fallback to initial guess if optimization fails
        centers = centers_init.copy()
        radii = radii_init.copy()
        
    # Ensure non-negative radii and strict boundary adherence
    radii = np.maximum(radii, 0.0)
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    return centers, radii, np.sum(radii)