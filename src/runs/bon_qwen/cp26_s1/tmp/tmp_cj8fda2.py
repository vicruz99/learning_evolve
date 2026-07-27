import numpy as np
from scipy.optimize import minimize

def compute_sum_radii(positions):
    """
    Computes the sum of maximal radii for a given set of circle centers.
    Each radius is constrained by the square boundaries and other circles.
    """
    n = positions.shape[0]
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    r = np.minimum(np.minimum(positions[:, 0], 1.0 - positions[:, 0]),
                   np.minimum(positions[:, 1], 1.0 - positions[:, 1]))
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Radius constrained by nearest neighbor: dist / 2
    min_dist = np.min(dists / 2.0, axis=1)
    r = np.minimum(r, min_dist)
    
    return np.sum(r)

def objective_func(pos_flat):
    """Objective function for optimization: maximize sum of radii -> minimize negative sum."""
    pos = pos_flat.reshape((26, 2))
    pos = np.clip(pos, 0.0, 1.0)
    return -compute_sum_radii(pos)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    best_score = -np.inf
    best_pos_flat = None
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    candidates = []
    
    # 1. Hexagonal-like lattice
    h, w = 5, 6
    gy, gx = np.meshgrid(np.linspace(0.08, 0.92, h), np.linspace(0.08, 0.92, w))
    for i in range(0, w, 2):
        gx[:, i] += 0.04
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:n]
    candidates.append(pts.ravel())
    
    # 2. Uniform grid with center point
    gy2, gx2 = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    pts2 = np.column_stack([gx2.ravel(), gy2.ravel()])
    pts2 = np.vstack([pts2, [0.5, 0.5]])
    candidates.append(pts2.ravel())
    
    # 3-8. Random uniform starts
    for _ in range(6):
        candidates.append(rng.uniform(0.0, 1.0, size=2 * n))
        
    # Optimize from each start
    for x0 in candidates:
        try:
            # Powell is derivative-free and robust for non-smooth objectives
            res = minimize(objective_func, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 4000, 'ftol': 1e-10, 'xtol': 1e-10})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pos_flat = res.x.copy()
        except Exception:
            continue

    # Ensure we have a result
    if best_pos_flat is None:
        best_pos_flat = rng.uniform(0.0, 1.0, size=2 * n)
        
    best_positions = best_pos_flat.reshape((n, 2))
    
    # Strictly enforce interior bounds to satisfy validation tolerances
    best_positions = np.clip(best_positions, 1e-6, 1.0 - 1e-6)
    
    # Recompute exact radii from final positions
    radii = np.minimum(np.minimum(best_positions[:, 0], 1.0 - best_positions[:, 0]),
                       np.minimum(best_positions[:, 1], 1.0 - best_positions[:, 1]))
    diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists / 2.0, axis=1)
    radii = np.minimum(radii, min_dist)
    
    total_sum = np.sum(radii)
    return best_positions, radii, total_sum