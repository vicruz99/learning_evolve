import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n, P):
    """Compute penalized objective: negative sum of radii + boundary/overlap penalties"""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Base objective: maximize sum of radii
    val = -np.sum(r)
    
    # Boundary penalties (quadratic)
    val += P * np.sum(np.maximum(0.0, r - c[:, 0])**2)
    val += P * np.sum(np.maximum(0.0, c[:, 0] + r - 1.0)**2)
    val += P * np.sum(np.maximum(0.0, r - c[:, 1])**2)
    val += P * np.sum(np.maximum(0.0, c[:, 1] + r - 1.0)**2)
    
    # Overlap penalties
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    iu, ju = np.triu_indices(n, k=1)
    viol = r_sum[iu, ju] - dists[iu, ju]
    viol_pos = np.maximum(0.0, viol)
    
    # Quadratic penalty for overlaps, plus small linear term for steeper gradient
    val += P * np.sum(viol_pos**2)
    val += P * 0.05 * np.sum(viol_pos)
    
    return val

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Test multiple initial radii to avoid local minima
    for init_r in [0.08, 0.09, 0.10]:
        centers = np.zeros((n, 2))
        radii = np.full(n, init_r)
        
        # Hexagonal-ish initialization pattern
        idx = 0
        for i in range(5):
            y = 0.1 + i * 0.2
            for j in range(6):
                x = 0.1 + j * 0.18 + (0.09 if i % 2 else 0.0)
                if idx < n:
                    centers[idx] = [x, y]
                    idx += 1
                    
        x0 = np.concatenate([centers.flatten(), radii])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        res = minimize(compute_objective, x0, args=(n, 10000.0), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10})
        
        curr_radii = res.x[2*n:]
        actual_sum = np.sum(curr_radii)
        
        if actual_sum > best_sum:
            best_sum = actual_sum
            best_centers = res.x[:2*n].reshape(n, 2)
            best_radii = curr_radii.copy()
            
    centers = best_centers
    radii = best_radii
    
    # Post-processing: enforce strict constraints within numerical tolerance
    for i in range(n):
        r = radii[i]
        if centers[i, 0] - r < 0: radii[i] = centers[i, 0]
        if centers[i, 0] + r > 1: radii[i] = 1.0 - centers[i, 0]
        if centers[i, 1] - r < 0: radii[i] = centers[i, 1]
        if centers[i, 1] + r > 1: radii[i] = 1.0 - centers[i, 1]
        
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < radii[i] + radii[j]:
                excess = radii[i] + radii[j] - d
                radii[i] -= excess / 2
                radii[j] -= excess / 2
                
    radii = np.maximum(radii, 0.0)
    return centers, radii, np.sum(radii)