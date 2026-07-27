import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii, weight):
    """Compute total penalty for boundary violations and overlaps."""
    n = len(radii)
    pen = 0.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Boundary penalties
        if x - r < 0:
            pen += weight * (x - r)**2
        if x + r > 1:
            pen += weight * (x + r - 1)**2
        if y - r < 0:
            pen += weight * (y - r)**2
        if y + r > 1:
            pen += weight * (y + r - 1)**2
            
        # Overlap penalties
        for j in range(i + 1, n):
            dx = x - centers[j, 0]
            dy = y - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            req = r + radii[j]
            
            if dist < req:
                pen += weight * (dist - req)**2
                
    return pen

def objective(params, n, weight):
    """Objective function: minimize -sum(radii) + penalty."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    return -np.sum(radii) + compute_penalty(centers, radii, weight)

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    np.random.seed(42)
    n = 26
    
    # Bounds: coordinates in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    best_params = None
    best_val = 1e9
    
    # Multi-restart optimization
    for _ in range(10):
        # Start with valid, small circles well inside the square
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        radii = np.full(n, 0.04)
        params = np.concatenate([centers.flatten(), radii])
        
        # Stage 1: Moderate penalty to find feasible configuration
        res1 = minimize(objective, params, args=(n, 100), method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12})
        
        # Stage 2: High penalty to tightly maximize radii while enforcing constraints
        res2 = minimize(objective, res1.x, args=(n, 5000), method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-12})
                        
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x
            
    centers = best_params[:2*n].reshape(n, 2)
    radii = best_params[2*n:]
    
    # Final strict boundary clamping
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    total_sum = np.sum(radii)
    return centers, radii, total_sum