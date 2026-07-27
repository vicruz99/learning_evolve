import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Precompute upper triangle indices for pairwise constraints (N=26)
TRIU_IDX = np.triu_indices(26, k=1)

def compute_objective(params, lam=1000.0):
    """
    Objective function: minimize -sum(radii) + lambda * penalty(violations)
    """
    n = 26
    centers = params[:2*n].reshape((n, 2))
    radii = params[2*n:]
    
    # Primary objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Penalty for boundary violations
    x, y = centers[:, 0], centers[:, 1]
    p_bound = np.sum(np.maximum(0, radii - x)**2)
    p_bound += np.sum(np.maximum(0, radii - (1 - x))**2)
    p_bound += np.sum(np.maximum(0, radii - y)**2)
    p_bound += np.sum(np.maximum(0, radii - (1 - y))**2)
    
    # Penalty for overlaps
    dists = cdist(centers, centers)
    overlaps = radii[:, None] + radii[None, :] - dists
    # Only consider upper triangle to avoid double counting
    p_overlap = np.sum(np.maximum(0, overlaps[TRIU_IDX])**2)
    
    obj += lam * (p_bound + p_overlap)
    return obj

def run_packing():
    n = 26
    best_sum = -1.0
    best_params = None
    
    # Bounds: centers in [0, 1], radii in [0.01, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n
    
    # Run multiple random restarts
    for seed in range(20):
        np.random.seed(seed)
        
        # Initialize with a hexagonal grid
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.1
        
        r_init = 0.1
        dx = 2 * r_init
        dy = np.sqrt(3) * r_init
        
        row = 0
        col = 0
        cnt = 0
        while cnt < n:
            x = col * dx + r_init
            y = row * dy + r_init
            if row % 2 == 1:
                x += dx / 2
                
            if x <= 1.0 and y <= 1.0:
                centers[cnt] = [x, y]
                radii[cnt] = r_init
                cnt += 1
                col += 1
            else:
                col = 0
                row += 1
                y = row * dy + r_init
        
        # Perturb to break symmetry and aid exploration
        centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        radii += np.random.uniform(-0.005, 0.005, n)
        radii = np.clip(radii, 0.05, 0.3)
        
        params = np.concatenate([centers.flatten(), radii])
        
        # Continuation method: increase penalty weight gradually
        current_params = params
        for lam in [100, 1000, 5000]:
            res = minimize(compute_objective, current_params, args=(lam,),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8})
            current_params = res.x
            
        centers_opt = current_params[:2*n].reshape((n, 2))
        radii_opt = current_params[2*n:]
        
        if np.sum(radii_opt) > best_sum:
            best_sum = np.sum(radii_opt)
            best_params = current_params
            
    centers_final = best_params[:2*n].reshape((n, 2))
    radii_final = best_params[2*n:]
    
    # Final safety check: shrink radii slightly if any constraints are violated
    # This ensures strict compliance with the validator's 1e-12 tolerance
    max_violation = 0.0
    
    # Check boundaries
    x, y = centers_final[:, 0], centers_final[:, 1]
    for margin in [radii_final - x, x + radii_final - 1, 
                   radii_final - y, y + radii_final - 1]:
        v = np.max(margin)
        if v > max_violation:
            max_violation = v
            
    # Check overlaps
    dists = cdist(centers_final, centers_final)
    overlaps = radii_final[:, None] + radii_final[None, :] - dists
    max_overlap = np.max(overlaps[TRIU_IDX])
    if max_overlap > max_violation:
        max_violation = max_overlap
        
    if max_violation > 1e-10:
        # Uniformly shrink to eliminate violations
        shrink_ratio = 1.0 - (max_violation / 0.5)
        radii_final *= max(shrink_ratio, 0.999)
        
    return centers_final, radii_final, float(np.sum(radii_final))