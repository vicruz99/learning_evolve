import numpy as np
from scipy.optimize import differential_evolution, minimize

def calculate_sum_radii(centers):
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    dist_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    diff = centers[np.newaxis, :, :] - centers[:, np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    radii = np.minimum(dist_bound, 0.5 * min_dists)
    return np.sum(radii)

def objective(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    return -calculate_sum_radii(centers)

def generate_hex_grid(n):
    rows = int(np.sqrt(n * 1.1547))
    centers = []
    y = 0.1
    while len(centers) < n:
        x = 0.1
        while x <= 0.9 and len(centers) < n:
            centers.append([x, y])
            x += np.sqrt(3) * 0.1
        y += 0.15
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0, 1)] * (n * 2)
    
    centers_init = generate_hex_grid(n).flatten()
    bounds_de = [(0.05, 0.95) for _ in range(n * 2)]
    
    result_de = differential_evolution(
        objective, bounds_de, popsize=15, maxiter=50, 
        mutation=(0.5, 1.0), recombination=0.9, seed=42, polish=False
    )
    
    result_refined = minimize(
        objective, result_de.x, method='Nelder-Mead', 
        options={'maxiter': 50000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    centers = result_refined.x.reshape(-1, 2)
    
    # Final radius calculation
    x, y = centers[:, 0], centers[:, 1]
    dist_bound = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    diff = centers[np.newaxis, :, :] - centers[:, np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    radii = np.minimum(dist_bound, 0.5 * min_dists)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii