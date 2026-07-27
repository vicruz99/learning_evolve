import numpy as np
from scipy.optimize import differential_evolution, minimize

def compute_radii(centers):
    """
    Given an array of centers, compute the maximal non-overlapping radii
    that fit within the unit square.
    """
    n = centers.shape[0]
    # Compute pairwise Euclidean distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Ignore self-distances by setting diagonal to infinity
    dists[np.eye(n, dtype=bool)] = np.inf
    
    # Minimum distance to any other circle center
    min_pair_dists = np.min(dists, axis=1)
    
    # Minimum distance to the four walls
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # The largest valid radius is half the smallest of these constraints
    radii = 0.5 * np.minimum(min_pair_dists, wall_dists)
    return radii

def objective(x):
    """Negative sum of radii for optimization (minimization convention)"""
    centers = x.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Step 1: Global exploration using Differential Evolution
    # Handles non-smooth objective and finds a high-quality basin
    de_res = differential_evolution(
        objective, 
        bounds, 
        popsize=25, 
        maxiter=600, 
        seed=42, 
        tol=1e-6, 
        polish=False  # We handle local search manually for better control
    )
    
    # Step 2: Local refinement using Nelder-Mead simplex
    # Derivative-free, robust to the min() non-differentiabilities
    lm_res = minimize(
        objective, 
        de_res.x, 
        method='Nelder-Mead', 
        options={'maxiter': 25000, 'xatol': 1e-8, 'fatol': 1e-9}
    )
    
    # Extract and format results
    centers = lm_res.x.reshape(-1, 2)
    
    # Strict boundary enforcement to satisfy validator tolerances
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    # Recompute radii one last time to ensure consistency with final clipped positions
    radii = compute_radii(centers)
    total_sum = float(np.sum(radii))
    
    return centers, radii, total_sum