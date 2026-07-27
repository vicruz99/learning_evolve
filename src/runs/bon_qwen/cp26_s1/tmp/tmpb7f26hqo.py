import numpy as np
from scipy.optimize import minimize

def compute_loss(centers_flat, r):
    """
    Computes the sum of squared violations of packing constraints.
    centers_flat: 1D array of length 52 (26 circles * 2 coords)
    r: float, radius of circles
    """
    centers = centers_flat.reshape(26, 2)
    n = 26
    loss = 0.0
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Violation if x < r or x > 1-r
    dx_low = np.maximum(r - centers[:, 0], 0)
    dx_high = np.maximum(centers[:, 0] - (1.0 - r), 0)
    dy_low = np.maximum(r - centers[:, 1], 0)
    dy_high = np.maximum(centers[:, 1] - (1.0 - r), 0)
    
    loss += np.sum(dx_low**2) + np.sum(dx_high**2)
    loss += np.sum(dy_low**2) + np.sum(dy_high**2)
    
    # Pairwise constraints: distance >= 2r
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Violation if dist < 2r
    violations = np.maximum(2*r - dists, 0)
    
    # Ignore self-distances (diagonal)
    violations[np.diag_indices_from(violations)] = 0.0
    
    loss += np.sum(violations**2)
    
    return loss

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    Returns centers, radii, and sum of radii.
    """
    n_circles = 26
    bounds = [(0.0, 1.0)] * (2 * n_circles)
    
    # Initial guess: Random distribution
    np.random.seed(123)
    current_centers = np.random.rand(n_circles, 2).flatten()
    
    # Start with a small radius that is definitely feasible
    r_current = 0.04
    
    # First, relax the initial centers at small r
    res = minimize(compute_loss, current_centers, args=(r_current,), method='L-BFGS-B', 
                   bounds=bounds, options={'ftol': 1e-10, 'maxiter': 1000})
    current_centers = res.x
    
    # Adaptive step-up search to maximize r
    step = 0.01
    max_r = 0.15 # Upper bound estimate
    
    while step > 1e-5:
        r_next = min(r_current + step, max_r)
        
        # Optimize centers for the new radius
        res = minimize(compute_loss, current_centers, args=(r_next,), method='L-BFGS-B',
                       bounds=bounds, options={'ftol': 1e-10, 'maxiter': 2000})
        
        # Check if constraints are satisfied sufficiently
        if res.fun < 1e-8:
            r_current = r_next
            current_centers = res.x
        else:
            step /= 2.0
            
    # Final validation pass to ensure tight constraints
    # Sometimes local minima might have small residual loss.
    # We can try one more refinement at the found r_current
    res_final = minimize(compute_loss, current_centers, args=(r_current,), method='L-BFGS-B',
                         bounds=bounds, options={'ftol': 1e-12, 'maxiter': 3000})
    
    final_centers = res_final.x.reshape(n_circles, 2)
    final_radii = np.full(n_circles, r_current)
    sum_radii = np.sum(final_radii)
    
    # Clamp centers to [0,1] strictly for safety
    final_centers = np.clip(final_centers, 0.0, 1.0)
    
    return final_centers, final_radii, sum_radii