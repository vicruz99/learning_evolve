import numpy as np
from scipy.optimize import minimize

def compute_constraints(x, n):
    """Compute all boundary and non-overlap constraints as fun(x) >= 0."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    cons = []
    # Boundary constraints: cx >= r, cx + r <= 1, cy >= r, cy + r <= 1
    cons.extend(cx - r)
    cons.extend(1.0 - cx - r)
    cons.extend(cy - r)
    cons.extend(1.0 - cy - r)
    
    # Vectorized pairwise non-overlap constraints
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist = np.hypot(cx_diff, cy_diff)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract upper triangle (i < j)
    i, j = np.triu_indices(n, k=1)
    overlaps = dist[i, j] - r_sum[i, j]
    cons.extend(overlaps)
    
    return np.array(cons)

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Grid initialization
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cx_grid = np.linspace(0.15, 0.85, cols)
    cy_grid = np.linspace(0.15, 0.85, rows)
    cx_init, cy_init = [], []
    for y in cy_grid:
        for x in cx_grid:
            if len(cx_init) >= n:
                break
            cx_init.append(x)
            cy_init.append(y)
    configs.append((cx_init[:n], cy_init[:n]))
    
    # 2. Hexagonal lattice initialization
    cx_hex, cy_hex = [], []
    r_est = 0.095
    for row in range(6):
        for col in range(6):
            if len(cx_hex) >= n:
                break
            x = 0.12 + col * 2 * r_est + (row % 2) * r_est
            y = 0.12 + row * np.sqrt(3) * r_est
            if x < 0.88 and y < 0.88:
                cx_hex.append(x)
                cy_hex.append(y)
    if len(cx_hex) >= n:
        configs.append((cx_hex[:n], cy_hex[:n]))
        
    for idx, (cx0, cy0) in enumerate(configs):
        np.random.seed(idx * 13 + 7)
        # Add small perturbations to escape symmetry
        cx0 = np.array(cx0) + np.random.randn(n) * 0.008
        cy0 = np.array(cy0) + np.random.randn(n) * 0.008
        cx0 = np.clip(cx0, 0.05, 0.95)
        cy0 = np.clip(cy0, 0.05, 0.95)
        r0 = np.full(n, 0.085)
        x0 = np.concatenate([cx0, cy0, r0])
        
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        def obj(x):
            return -np.sum(x[2*n:])
            
        cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
        
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
        
        cx_opt = res.x[:n]
        cy_opt = res.x[n:2*n]
        r_opt = res.x[2*n:]
        
        # Post-processing: ensure strict validity
        r_opt = np.maximum(r_opt, 0.0)
        cx_opt = np.clip(cx_opt, r_opt, 1.0 - r_opt)
        cy_opt = np.clip(cy_opt, r_opt, 1.0 - r_opt)
        
        current_sum = np.sum(r_opt)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = np.column_stack([cx_opt, cy_opt])
            best_radii = r_opt
            
    return best_centers, best_radii, best_sum