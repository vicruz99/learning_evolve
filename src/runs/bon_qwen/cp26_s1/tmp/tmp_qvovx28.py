import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii, i_idx, j_idx):
    """
    Computes a penalty value for circle overlaps and boundary violations.
    Returns 0.0 if the configuration is valid.
    """
    n = len(radii)
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Overlap penalty (upper triangle)
    overlaps = (radii[i_idx] + radii[j_idx]) - dists[i_idx, j_idx]
    pen = np.sum(np.maximum(0.0, overlaps)**2)
    
    # Boundary penalty
    x = centers[:, 0]
    y = centers[:, 1]
    pen += np.sum(np.maximum(0.0, radii - x)**2)
    pen += np.sum(np.maximum(0.0, radii - (1.0 - x))**2)
    pen += np.sum(np.maximum(0.0, radii - y)**2)
    pen += np.sum(np.maximum(0.0, radii - (1.0 - y))**2)
    
    return pen

def packing_objective(vars_, radii, i_idx, j_idx):
    """Wrapper for scipy.optimize.minimize"""
    n = len(radii)
    centers = vars_[:2*n].reshape(n, 2)
    return compute_penalty(centers, radii, i_idx, j_idx)

def run_packing():
    np.random.seed(42)
    n = 26
    i_idx, j_idx = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Structured initial placement (grid-like)
    centers = np.zeros((n, 2))
    idx = 0
    for r in range(6):
        for c in range(5):
            if idx < n:
                centers[idx, 0] = (c + 0.5) / 5.0
                centers[idx, 1] = (r + 0.5) / 6.0
                idx += 1
    np.random.shuffle(centers)
    
    best_centers = centers.copy()
    
    # Pre-spread with a small radius to find a feasible baseline layout
    current_radii = np.ones(n) * 0.04
    res = minimize(packing_objective, best_centers.flatten(), 
                   args=(current_radii, i_idx, j_idx), 
                   bounds=bounds, method='L-BFGS-B', options={'maxiter': 500})
    best_centers = res.x[:2*n].reshape(n, 2)
    
    # Binary search for the maximum sum of radii
    low, high = 0.0, 2.85  # Target is 2.636, upper bound gives room
    for step in range(40):
        mid = (low + high) / 2.0
        current_radii = np.ones(n) * (mid / n)
        
        success = False
        # Try multiple initializations to avoid local minima
        for restart in range(3):
            init_c = best_centers.copy()
            if restart > 0:
                init_c += np.random.randn(n, 2) * 0.02
                init_c = np.clip(init_c, 0.05, 0.95)
                
            res = minimize(packing_objective, init_c.flatten(), 
                           args=(current_radii, i_idx, j_idx), 
                           bounds=bounds, method='L-BFGS-B', options={'maxiter': 500})
            
            if res.fun < 1e-8:
                best_centers = res.x[:2*n].reshape(n, 2)
                success = True
                break
                
        if success:
            low = mid
        else:
            high = mid
            
    final_sum = low
    final_radii = np.ones(n) * (final_sum / n)
    
    # Final safety clip to ensure bounds are strictly respected
    best_centers = np.clip(best_centers, final_radii, 1.0 - final_radii)
    
    return best_centers, final_radii, final_sum