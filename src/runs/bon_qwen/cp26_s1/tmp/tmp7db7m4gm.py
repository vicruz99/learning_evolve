import numpy as np
from scipy.optimize import minimize

def compute_radii(centers):
    """
    Computes the maximum valid radius for each circle given their centers.
    Radii are constrained by the unit square boundaries and non-overlap with other circles.
    """
    n = centers.shape[0]
    
    # 1. Boundary constraints: distance to nearest edge
    # min(x, 1-x, y, 1-y)
    radii = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                       np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # 2. Circle-circle constraints: half the distance to the nearest neighbor
    # Compute pairwise distances efficiently using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distance by setting diagonal to infinity
    dists[np.arange(n), np.arange(n)] = np.inf
    
    # Minimum distance to any other circle
    min_dists = np.min(dists, axis=1)
    
    # Radius is limited by half the distance to the nearest circle
    radii = np.minimum(radii, min_dists / 2.0)
    
    # Ensure non-negative (should be naturally positive if centers are valid)
    radii = np.maximum(radii, 0.0)
    
    return radii

def objective_function(params):
    """
    Objective function to maximize the sum of radii.
    Returns negative sum because scipy minimizes.
    """
    n = 26
    centers = params.reshape(n, 2)
    
    # Enforce bounds strictly within the square
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    
    radii = compute_radii(centers)
    return -np.sum(radii)

def generate_hexagonal_packing(n):
    """
    Generates an initial configuration based on a hexagonal lattice,
    which is a dense packing structure.
    """
    centers = []
    r_guess = 0.105
    rows = 8
    
    for row in range(rows):
        y = r_guess + row * np.sqrt(3) * r_guess
        # Shift odd rows horizontally
        offset = np.sqrt(3) * r_guess / 2.0 if row % 2 == 1 else 0.0
        
        col = 0
        while True:
            x = r_guess + col * 2.0 * r_guess + offset
            if x > 1.0 - r_guess:
                break
            centers.append([x, y])
            if len(centers) >= n:
                return np.array(centers[:n])
            col += 1
            
    return np.array(centers[:n])

def generate_grid_packing(n):
    """
    Generates a uniform grid configuration.
    """
    centers = []
    # 5x5 grid spacing
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            centers.append([x, y])
            if len(centers) >= n:
                return np.array(centers[:n])
        if len(centers) >= n:
            return np.array(centers[:n])
            
    # If needed, fill with random points
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to run the packing optimization.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Hexagonal packing
    initial_configs.append(generate_hexagonal_packing(n))
    
    # 2. Grid packing
    initial_configs.append(generate_grid_packing(n))
    
    # 3. Random configurations
    for _ in range(4):
        initial_configs.append(np.random.rand(n, 2))
        
    # 4. Jittered hexagonal packing to escape symmetry
    hex_cfg = generate_hexagonal_packing(n)
    jittered = hex_cfg + np.random.normal(0, 0.02, hex_cfg.shape)
    jittered = np.clip(jittered, 0.0, 1.0)
    initial_configs.append(jittered)
    
    # Optimize from each starting point
    for init_centers in initial_configs:
        params = init_centers.flatten()
        
        # Use Nelder-Mead for non-smooth objective
        res = minimize(objective_function, params, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-7})
        
        curr_sum = -res.fun
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = res.x.reshape(n, 2)
            
    # Final cleanup and validation
    best_centers = np.clip(best_centers, 1e-9, 1.0 - 1e-9)
    best_radii = compute_radii(best_centers)
    
    return best_centers, best_radii, np.sum(best_radii)