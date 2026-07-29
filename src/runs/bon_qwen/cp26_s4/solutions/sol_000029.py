# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 55285a70) state=d37e9356 sum of radii=2.457530 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_radii(centers):
    """
    Calculates the maximum possible radii for circles given their centers.
    Radii are constrained by boundaries and pairwise distances.
    """
    N = centers.shape[0]
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    dist_to_boundary = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    
    # Vectorized pairwise distances
    # Calculate (x_i - x_j)^2 + (y_i - y_j)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    dists_sq[np.arange(N), np.arange(N)] = np.inf
    
    # Minimum distance to any other circle
    min_dists_sq = np.min(dists_sq, axis=1)
    min_dists = np.sqrt(min_dists_sq)
    
    # Constraint from pairwise distances is half the distance
    radii_pairwise = min_dists / 2.0
    
    # Final radius is the tighter of the two constraints
    return np.minimum(dist_to_boundary, radii_pairwise)

def objective(x):
    """
    Objective function for optimization.
    Minimizes negative sum of radii (maximizes sum of radii).
    """
    centers = x.reshape(-1, 2)
    radii = get_radii(centers)
    return -np.sum(radii)

def hex_grid_init(n_points):
    """Generates a hexagonal grid of centers suitable for 26 circles."""
    centers = []
    r_est = 0.11 # Estimate slightly larger to fill space
    dy = r_est * np.sqrt(3)
    y = r_est
    
    for row in range(8): # Enough rows to cover the square
        x = r_est
        # Offset odd rows to create hexagonal packing
        if row % 2 == 1:
            x = r_est + r_est / 2 # Shift by half diameter (approx r)
            
        while x + r_est <= 1.0:
            centers.append([x, y])
            x += 2 * r_est
            if len(centers) >= n_points:
                return np.array(centers[:n_points])
        y += dy
        
    # Fallback if grid didn't yield enough points
    while len(centers) < n_points:
        centers.append(np.random.rand(2) * 0.8 + 0.1)
    return np.array(centers[:n_points])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    best_sum = -1.0
    best_centers = None
    
    # Strategy 1: Hexagonal lattice initialization
    init_hex = hex_grid_init(N)
    res_hex = minimize(objective, init_hex.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
    if -res_hex.fun > best_sum:
        best_sum = -res_hex.fun
        best_centers = res_hex.x.reshape(-1, 2)
        
    # Strategy 2: Perturbed square grid (5x5 with extra)
    grid_points = []
    # 5x5 grid
    for r in range(5):
        for c in range(5):
            grid_points.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    # Add 6th point in a gap (center of a cell)
    grid_points.append([0.3, 0.3]) 
    # Select 26
    grid_points = np.array(grid_points[:N])
    # Perturb slightly to break symmetry
    grid_points += np.random.normal(0, 0.01, grid_points.shape)
    
    res_grid = minimize(objective, grid_points.flatten(), method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
    if -res_grid.fun > best_sum:
        best_sum = -res_grid.fun
        best_centers = res_grid.x.reshape(-1, 2)
        
    # Strategy 3: Random restarts for robustness
    for _ in range(5):
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        res_rand = minimize(objective, centers.flatten(), method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
        if -res_rand.fun > best_sum:
            best_sum = -res_rand.fun
            best_centers = res_rand.x.reshape(-1, 2)

    # Final local perturbation / Hill climbing refinement
    for _ in range(200):
        idx = np.random.randint(N)
        # Small random perturbation
        delta = np.random.normal(0, 0.005, 2)
        new_centers = best_centers.copy()
        new_centers[idx] += delta
        # Clamp to keep inside square with some margin
        new_centers = np.clip(new_centers, 0.01, 0.99)
        
        new_radii = get_radii(new_centers)
        if np.sum(new_radii) > best_sum:
            best_sum = np.sum(new_radii)
            best_centers = new_centers

    # Final calculation
    radii = get_radii(best_centers)
    return best_centers, radii, np.sum(radii)
