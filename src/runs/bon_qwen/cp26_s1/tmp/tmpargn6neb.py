import numpy as np
from scipy.optimize import minimize

def compute_penalty(vars_arr, n, lam):
    """Computes the penalty for constraint violations."""
    c = vars_arr[:2*n].reshape(n, 2)
    r = vars_arr[2*n:]
    pen = 0.0
    
    # Boundary penalties: ensure r <= x <= 1-r and r <= y <= 1-r
    pen += np.sum(np.maximum(r - c[:,0], 0)**2)
    pen += np.sum(np.maximum(r - (1 - c[:,0]), 0)**2)
    pen += np.sum(np.maximum(r - c[:,1], 0)**2)
    pen += np.sum(np.maximum(r - (1 - c[:,1]), 0)**2)
    
    # Overlap penalties: ensure dist >= r_i + r_j
    # Vectorized distance computation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlaps = np.maximum(r_sum - dist, 0)
    
    # Sum over upper triangle only to avoid double counting
    triu_idx = np.triu_indices(n, k=1)
    pen += np.sum(overlaps[triu_idx]**2)
    
    return lam * pen

def objective(vars_arr, n):
    """Negative sum of radii (we minimize this to maximize sum)."""
    return -np.sum(vars_arr[2*n:])

def obj_func(vars_arr, n, lam):
    """Combined objective and penalty function."""
    return objective(vars_arr, n) + compute_penalty(vars_arr, n, lam)

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.08
    
    # Initialize with a hexagonal-like pattern for better starting configuration
    row_counts = [6, 5, 6, 5, 4]
    idx = 0
    y = 0.1
    for i, cnt in enumerate(row_counts):
        x_start = (1.0 - cnt * 0.16) / 2.0 + 0.08
        if i % 2 == 1:
            x_start += 0.08  # Offset for hexagonal staggering
        for j in range(cnt):
            if idx < n:
                centers[idx] = [x_start + j * 0.16, y]
                idx += 1
        y += 0.14  # Vertical spacing approx 0.08 * sqrt(3)
        
    # Flatten to 1D array: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: x,y in [0.01, 0.99], r in [0.001, 0.5]
    bounds = [(0.01, 0.99)] * (2*n) + [(0.001, 0.5)] * n
    
    # Stage 1: Coarse optimization with moderate penalty
    res1 = minimize(obj_func, x0, args=(n, 1000.0), method='L-BFGS-B', bounds=bounds, 
                    options={'maxiter': 2000, 'ftol': 1e-10})
    
    # Stage 2: Fine optimization with high penalty to strictly enforce constraints
    res2 = minimize(obj_func, res1.x, args=(n, 10000.0), method='L-BFGS-B', bounds=bounds, 
                    options={'maxiter': 2000, 'ftol': 1e-10})
    
    c_final = res2.x[:2*n].reshape(n, 2)
    r_final = res2.x[2*n:]
    
    # Post-processing to guarantee strict validity for the checker
    # 1. Enforce boundary constraints
    for i in range(n):
        r_final[i] = min(r_final[i], c_final[i,0], 1-c_final[i,0], c_final[i,1], 1-c_final[i,1])
        
    # 2. Resolve any remaining overlaps by shrinking radii equally
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((c_final[i] - c_final[j])**2))
            if d < r_final[i] + r_final[j] - 1e-12:
                overlap = r_final[i] + r_final[j] - d
                r_final[i] -= overlap / 2.0
                r_final[j] -= overlap / 2.0
                
    # 3. Final boundary clamp and non-negativity check
    for i in range(n):
        r_final[i] = max(0.0, min(r_final[i], c_final[i,0], 1-c_final[i,0], c_final[i,1], 1-c_final[i,1]))
        
    return c_final, r_final, np.sum(r_final)