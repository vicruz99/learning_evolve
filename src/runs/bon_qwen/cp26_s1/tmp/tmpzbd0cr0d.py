import numpy as np
from scipy.optimize import minimize
import functools

def objective(vars):
    n = len(vars) // 3
    radii = vars[2::3]
    return -np.sum(radii)

def boundary_x_min(v, i):
    return v[3*i] - v[3*i+2]

def boundary_x_max(v, i):
    return 1.0 - (v[3*i] + v[3*i+2])

def boundary_y_min(v, i):
    return v[3*i+1] - v[3*i+2]

def boundary_y_max(v, i):
    return 1.0 - (v[3*i+1] + v[3*i+2])

def radius_pos(v, i):
    return v[3*i+2]

def overlap(v, i, j):
    c1 = v[3*i:3*i+2]
    c2 = v[3*j:3*j+2]
    r1 = v[3*i+2]
    r2 = v[3*j+2]
    d2 = np.sum((c1 - c2)**2)
    return d2 - (r1 + r2)**2

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    r_init = 0.09
    
    # Hexagonal initialization strategy
    # Arranges circles in a staggered grid to mimic optimal dense packing
    dx = 0.2
    dy = np.sqrt(3)/2 * dx
    idx = 0
    for row in range(7):
        y = 0.05 + row * dy
        x_start = 0.05 + (row % 2) * (dx / 2)
        x = x_start
        while idx < n and x + r_init < 0.95:
            centers[idx] = [x, y]
            idx += 1
            x += dx
        if idx == n:
            break
            
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    radii = np.full(n, r_init)
    x0 = np.concatenate([centers.flatten(), radii])
    
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': functools.partial(boundary_x_min, i=i)})
        cons.append({'type': 'ineq', 'fun': functools.partial(boundary_x_max, i=i)})
        cons.append({'type': 'ineq', 'fun': functools.partial(boundary_y_min, i=i)})
        cons.append({'type': 'ineq', 'fun': functools.partial(boundary_y_max, i=i)})
        cons.append({'type': 'ineq', 'fun': functools.partial(radius_pos, i=i)})
        
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': functools.partial(overlap, i=i, j=j)})
            
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, None) for _ in range(n)]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10, 'iprint': -1})
    
    final_vars = res.x
    final_centers = final_vars[:2*n].reshape(n, 2)
    final_radii = final_vars[2::3]
    
    # Post-processing to ensure strict validity against numerical errors
    final_radii = np.maximum(final_radii, 1e-9)
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, np.sum(final_radii)