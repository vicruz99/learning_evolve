import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal-like grid pattern
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        n_in_row = 6 if row % 2 == 0 else 5
        if idx + n_in_row > n:
            n_in_row = n - idx
        
        y_base = (row + 1) * 0.2
        x_start = 0.5 - (n_in_row - 1) * 0.08
        
        for col in range(n_in_row):
            centers[idx, 0] = x_start + col * 0.16
            centers[idx, 1] = y_base + (0.08 if row % 2 != 0 else 0.0)
            idx += 1
            
    radii = np.full(n, 0.06)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # 2. Objective function with penalty terms
    def objective(x, penalty_weight):
        c = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        
        loss = -np.sum(r)
        pen = 0.0
        
        # Boundary penalty
        for i in range(n):
            for k in range(2):
                if c[i, k] < r[i]:
                    pen += (r[i] - c[i, k])**2
                if c[i, k] > 1.0 - r[i]:
                    pen += (c[i, k] - (1.0 - r[i]))**2
                    
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                if d < r[i] + r[j]:
                    pen += (r[i] + r[j] - d)**2
                    
        return loss + penalty_weight * pen

    bnds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    # 3. Sequential Penalty Optimization
    x = x0
    penalty_weights = [100, 1000, 5000]
    
    for pw in penalty_weights:
        # Define wrapper to pass penalty weight without lambda
        def obj_wrapper(val):
            return objective(val, pw)
            
        res = minimize(obj_wrapper, x, method='L-BFGS-B', bounds=bnds, 
                       options={'maxiter': 500, 'ftol': 1e-10})
        x = res.x
        
    # 4. Extract and clean results
    opt_centers = x[:2*n].reshape(n, 2)
    opt_radii = x[2*n:]
    
    # Ensure radii are non-negative
    opt_radii = np.maximum(opt_radii, 0.0)
    
    # Clamp centers to strictly satisfy boundary constraints
    for i in range(n):
        opt_centers[i, 0] = np.clip(opt_centers[i, 0], opt_radii[i], 1.0 - opt_radii[i])
        opt_centers[i, 1] = np.clip(opt_centers[i, 1], opt_radii[i], 1.0 - opt_radii[i])
        
    total_sum = np.sum(opt_radii)
    return opt_centers, opt_radii, total_sum