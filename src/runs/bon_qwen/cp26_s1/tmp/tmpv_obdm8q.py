import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def all_constraints(x):
    """
    Computes all inequality constraints for the circle packing problem.
    Returns an array of constraint values, all of which must be >= 0.
    """
    n = N_CIRCLES
    r = x[:n]
    cx = x[n:2*n]
    cy = x[2*n:3*n]
    
    # Boundary constraints: cx >= r, 1-cx >= r, cy >= r, 1-cy >= r
    b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # Use broadcasting for efficiency
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum_sq = (r[:, None] + r[None, :])**2
    overlap = dist_sq - r_sum_sq
    
    # Extract only upper triangular part (i < j)
    idx = np.triu_indices(n, k=1)
    o_vals = overlap[idx]
    
    return np.concatenate([b, o_vals])

def run_packing():
    n = N_CIRCLES
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Prepare initial configurations
    starts = []
    
    # 1. 5x5 grid + center circle
    r1 = np.full(n, 0.1)
    c1 = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            c1[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
            idx += 1
    c1[25] = [0.5, 0.5]
    r1[25] = 0.05
    starts.append(np.concatenate([r1, c1.flatten()]))
    
    # 2. Random initialization
    np.random.seed(42)
    r2 = np.random.uniform(0.05, 0.12, n)
    c2 = np.random.uniform(0.2, 0.8, (n, 2))
    starts.append(np.concatenate([r2, c2.flatten()]))
    
    # 3. Hexagonal-ish arrangement
    r3 = np.full(n, 0.09)
    c3 = np.zeros((n, 2))
    idx = 0
    for row in range(6):
        num_circles = 5 if row % 2 == 0 else 4
        y = 0.15 + row * 0.14
        for col in range(num_circles):
            x = 0.15 + col * 0.2 + (row % 2) * 0.1
            if idx < n:
                c3[idx] = [x, y]
                idx += 1
    starts.append(np.concatenate([r3, c3.flatten()]))

    bounds = [(0.0, 0.5)] * n + [(0.0, 1.0)] * (2*n)
    cons = {'type': 'ineq', 'fun': all_constraints}
    
    for x0 in starts:
        def obj(x):
            return -np.sum(x[:n])
            
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            
            r_cand = res.x[:n]
            cx_cand = res.x[n:2*n]
            cy_cand = res.x[2*n:3*n]
            
            # Quick feasibility check with small tolerance
            valid = True
            if np.any(r_cand < 0) or np.any(r_cand > 0.5):
                valid = False
            else:
                # Check boundaries
                if np.any(cx_cand < r_cand - 1e-7) or np.any(cx_cand > 1 - r_cand + 1e-7):
                    valid = False
                if np.any(cy_cand < r_cand - 1e-7) or np.any(cy_cand > 1 - r_cand + 1e-7):
                    valid = False
                    
                # Check overlaps
                dx = cx_cand[:, None] - cx_cand[None, :]
                dy = cy_cand[:, None] - cy_cand[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r_cand[:, None] + r_cand[None, :]
                idx = np.triu_indices(n, k=1)
                if np.any(dist[idx] < min_dist[idx] - 1e-7):
                    valid = False

            if valid:
                sum_r = np.sum(r_cand)
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_radii = r_cand.copy()
                    best_centers = np.column_stack((cx_cand, cy_cand))
        except Exception:
            continue
            
    # Fallback if optimization completely fails (should not happen)
    if best_centers is None:
        best_radii = starts[0][:n]
        best_centers = starts[0][n:].reshape(-1, 2)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum