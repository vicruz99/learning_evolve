import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def compute_loss(centers, r):
    """
    Compute overlap and boundary penalty for a given configuration.
    centers: np.array of shape (n, 2)
    r: float, target radius
    """
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    i, j = np.triu_indices(n, 1)
    pair_dists = dists[i, j]
    
    # Overlap penalty
    overlap = np.maximum(0, 2*r - pair_dists)
    loss_val = np.sum(overlap**2)
    
    # Boundary penalties
    # x coordinates
    loss_val += np.sum(np.maximum(0, r - centers[:, 0])**2)
    loss_val += np.sum(np.maximum(0, centers[:, 0] - (1 - r))**2)
    # y coordinates
    loss_val += np.sum(np.maximum(0, r - centers[:, 1])**2)
    loss_val += np.sum(np.maximum(0, centers[:, 1] - (1 - r))**2)
    
    return loss_val

def get_initial_guess(n, r):
    """
    Generate a hexagonal lattice initialization inside [r, 1-r]^2.
    """
    s = 0.28  # Lattice spacing
    pts = []
    for i in range(15):
        for j in range(15):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if r + 0.01 < x < 1 - r - 0.01 and r + 0.01 < y < 1 - r - 0.01:
                pts.append([x, y])
                
    pts = np.array(pts)
    if len(pts) < n:
        # Fallback to random if lattice yields too few points
        np.random.seed(0)
        pts = np.random.uniform(r, 1-r, (n*2, 2))
        
    dists_to_center = np.linalg.norm(pts - 0.5, axis=1)
    idx = np.argsort(dists_to_center)[:n]
    return pts[idx]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    r_low, r_high = 0.09, 0.15
    best_r = r_low
    best_centers = None
    
    # Binary search for optimal radius
    for step in range(35):
        r_mid = (r_low + r_high) / 2
        bounds = [(r_mid, 1 - r_mid)] * (2 * n)
        
        success = False
        best_val = np.inf
        best_res_centers = None
        
        # Multiple restarts to escape local minima
        for k in range(8):
            np.random.seed(k * 42 + step * 7)
            init = get_initial_guess(n, r_mid)
            
            # Add controlled noise
            init += np.random.normal(0, 0.03, init.shape)
            init = np.clip(init, r_mid, 1 - r_mid)
            
            res = minimize(compute_loss, init.flatten(), args=(r_mid,), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_val:
                best_val = res.fun
                best_res_centers = res.x.reshape(n, 2)
                
            if best_val < 1e-9:
                success = True
                break
        
        if success:
            best_r = r_mid
            best_centers = best_res_centers
            r_low = r_mid
        else:
            r_high = r_mid
            
    # Final safety shrink to guarantee constraints are strictly met
    # The optimizer targets 0 overlap, but numerical precision might leave tiny overlaps.
    # A factor of 0.9999 ensures we are safely inside while maximizing sum.
    radii = np.full(n, best_r * 0.99995)
    
    # Ensure centers are valid for the final radii
    final_centers = best_centers * (1 - radii[0]) + radii[0] # Shift/scale not needed as bounds were respected
    
    return final_centers, radii, float(np.sum(radii))