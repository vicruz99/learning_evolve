import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, n, lam, iu, ju):
    """Compute penalized objective function for circle packing."""
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    # Primary objective: maximize sum of radii
    loss = -np.sum(r)
    
    # Boundary margins: x-r, 1-x-r, y-r, 1-y-r
    m1 = x - r
    m2 = (1.0 - x) - r
    m3 = y - r
    m4 = (1.0 - y) - r
    
    pen = np.sum(np.maximum(0.0, -m1)**2)
    pen += np.sum(np.maximum(0.0, -m2)**2)
    pen += np.sum(np.maximum(0.0, -m3)**2)
    pen += np.sum(np.maximum(0.0, -m4)**2)
    
    # Overlap margins: dist(i,j) - (r_i + r_j)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    margins_ov = dist[iu, ju] - r_sum[iu, ju]
    pen += np.sum(np.maximum(0.0, -margins_ov)**2)
    
    return loss + lam * pen

def run_packing():
    np.random.seed(42)
    n = 26
    iu, ju = np.triu_indices(n, k=1)
    
    # 1. Initialization: Staggered hexagonal grid
    cx = np.zeros(n)
    cy = np.zeros(n)
    cr = np.ones(n) * 0.05
    
    idx = 0
    for row in range(5):
        for col in range(6):
            if idx >= n: break
            cx[idx] = 0.12 + col * 0.14 + (row % 2) * 0.07
            cy[idx] = 0.12 + row * 0.17
            idx += 1
    while idx < n:
        cx[idx] = np.random.uniform(0.2, 0.8)
        cy[idx] = np.random.uniform(0.2, 0.8)
        idx += 1
        
    cx = np.clip(cx, 0.01, 0.99)
    cy = np.clip(cy, 0.01, 0.99)
    
    x0 = np.concatenate([cx, cy, cr])
    bounds = [(0.0, 1.0)]*n + [(0.0, 1.0)]*n + [(0.0, 0.5)]*n
    
    best_x = x0.copy()
    lam = 2000.0
    
    # 2. Continuation method with L-BFGS-B
    for stage in range(6):
        lam *= 1.6
        res = minimize(compute_loss, best_x, args=(n, lam, iu, ju), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-10})
        best_x = res.x
        
        # Small perturbation to escape local minima
        noise = np.random.normal(0, 0.0015, len(best_x))
        bounds_arr = np.array(bounds)
        best_x = np.clip(best_x + noise, bounds_arr[:, 0], bounds_arr[:, 1])
        
    cx_out = best_x[:n]
    cy_out = best_x[n:2*n]
    cr_out = best_x[2*n:3*n]
    
    # 3. Strict feasibility projection
    for _ in range(30):
        for i in range(n):
            limit = min(cx_out[i], 1.0 - cx_out[i], cy_out[i], 1.0 - cy_out[i])
            for j in range(n):
                if i == j: continue
                d = np.sqrt((cx_out[i]-cx_out[j])**2 + (cy_out[i]-cy_out[j])**2)
                limit = min(limit, d - cr_out[j])
            cr_out[i] = min(cr_out[i], max(0.0, limit - 1e-8))
            
        cx_out = np.clip(cx_out - cr_out, 0.0, 1.0) + cr_out
        cy_out = np.clip(cy_out - cr_out, 0.0, 1.0) + cr_out
        
    centers_out = np.column_stack([cx_out, cy_out])
    return centers_out, cr_out, float(np.sum(cr_out))