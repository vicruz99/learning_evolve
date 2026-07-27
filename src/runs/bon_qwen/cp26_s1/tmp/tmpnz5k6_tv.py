import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraint_vals(v):
    """
    Returns array of constraint values. All must be >= 0.
    v is ordered as [x0, y0, r0, x1, y1, r1, ..., xN, yN, rN]
    """
    n = N_CIRCLES
    c = []
    # Boundary constraints for each circle
    for i in range(n):
        c.append(v[3*i] - v[3*i+2])           # x - r >= 0
        c.append(1.0 - v[3*i] - v[3*i+2])     # 1 - x - r >= 0
        c.append(v[3*i+1] - v[3*i+2])         # y - r >= 0
        c.append(1.0 - v[3*i+1] - v[3*i+2])   # 1 - y - r >= 0
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            c.append(dist - (v[3*i+2] + v[3*j+2]))
            
    return np.array(c)

def run_packing():
    n = N_CIRCLES
    best_sum = 0.0
    best_res = None
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # Constraint dictionary for SLSQP
    cons = {'type': 'ineq', 'fun': constraint_vals}
    
    # Multiple restarts to escape local minima
    for seed in range(15):
        rng = np.random.default_rng(seed)
        
        # Initialize centers in the central region to avoid immediate boundary conflicts
        cx = rng.uniform(0.25, 0.75, n)
        cy = rng.uniform(0.25, 0.75, n)
        r_init = 0.03 * np.ones(n)
        
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = cx[i]
            x0[3*i+1] = cy[i]
            x0[3*i+2] = r_init[i]
            
        res = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success and -res.fun > best_sum:
            best_sum = -res.fun
            best_res = res
            
    # Extract and format results
    centers = np.column_stack((best_res.x[:n], best_res.x[n:2*n]))
    radii = np.maximum(0.0, best_res.x[2*n:3*n])  # Clamp tiny negatives from numerical noise
    
    return centers, radii, np.sum(radii)