# sol_000343 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=3a72e5a2 sum of radii=0.000117 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(vars_arr, P):
    """
    Computes the objective function: negative sum of radii + penalty for constraint violations.
    """
    x = vars_arr[0::3]
    y = vars_arr[1::3]
    r = vars_arr[2::3]
    
    # Boundary penalties: x >= r, x <= 1-r, y >= r, y <= 1-r
    bv = (np.maximum(r - x, 0.0)**2 + 
          np.maximum(r - (1.0 - x), 0.0)**2 + 
          np.maximum(r - y, 0.0)**2 + 
          np.maximum(r - (1.0 - y), 0.0)**2)
          
    # Pairwise overlap penalties: dist(c_i, c_j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    pv = np.maximum(dr - dist, 0.0)**2
    
    # Sum of squared violations weighted by P
    return -np.sum(r) + P * (np.sum(bv) + np.sum(pv))

def run_packing():
    np.random.seed(42)
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for [x, y, r] for each circle
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    for trial in range(12):
        # Initialize with a hexagonal grid pattern
        r_init = 0.08
        pts = []
        row_y = 0.0
        row_idx = 0
        while len(pts) < n:
            col_x = r_init if row_idx % 2 == 1 else 0.0
            while col_x + 2*r_init <= 1.0 and len(pts) < n:
                pts.append((col_x + r_init, row_y + r_init))
                col_x += 2 * r_init
            row_y += np.sqrt(3) * r_init
            row_idx += 1
            
        centers_init = np.array(pts[:n])
        
        # Add jitter to break symmetry and escape local minima
        centers_init += np.random.uniform(-0.02, 0.02, size=centers_init.shape)
        centers_init = np.clip(centers_init, 0.1, 0.9)
        
        radii_init = np.full(n, 0.05)
        
        # Flatten to optimization vector [x1, y1, r1, x2, y2, r2, ...]
        vars0 = np.zeros(3*n)
        for i in range(n):
            vars0[3*i] = centers_init[i, 0]
            vars0[3*i+1] = centers_init[i, 1]
            vars0[3*i+2] = radii_init[i]
            
        # Adaptive penalty method
        P = 10.0
        vars_curr = vars0.copy()
        for step in range(25):
            res = minimize(compute_loss, vars_curr, args=(P,), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            vars_curr = res.x
            
            # Check if constraints are sufficiently satisfied
            loss = res.fun
            neg_sum_r = -np.sum(vars_curr[2::3])
            penalty_term = loss - neg_sum_r
            
            if penalty_term < 1e-4:
                break
            P *= 5.0
            
        # Extract optimized parameters
        x = vars_curr[0::3]
        y = vars_curr[1::3]
        r = vars_curr[2::3]
        
        # Strict validation step
        valid = True
        for i in range(n):
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > 1 + 1e-6 or \
               y[i] - r[i] < -1e-6 or y[i] + r[i] > 1 + 1e-6:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_centers = np.column_stack((x, y))
                best_radii = r
                
    # Fallback grid packing in case optimizer fails (unlikely)
    if best_centers is None:
        step = 1.0 / 5
        best_centers = np.array([[i*step, j*step] for j in range(5) for i in range(5)][:26])
        best_radii = np.full(26, step/2 * 0.95)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
