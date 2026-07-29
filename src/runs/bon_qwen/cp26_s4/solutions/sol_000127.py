# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d5ce57f9) state=9fa7d2bc sum of radii=1.810787 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_radii(centers):
    """
    Computes the maximum valid radius for each circle given centers.
    Radii are constrained by distance to walls and distance to other circles.
    """
    n = centers.shape[0]
    
    # Distance to walls: min(x, 1-x, y, 1-y)
    x = centers[:, 0]
    y = centers[:, 1]
    dist_walls = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Distance to other circles
    # Calculate pairwise distances
    # diffs shape (n, n, 2)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists shape (n, n)
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    # Minimum distance to any other circle center
    min_dists = np.min(dists, axis=1)
    
    # Radius limited by half distance to nearest neighbor
    r_neighbors = 0.5 * min_dists
    
    # Final radius is the minimum of wall distance and neighbor distance
    radii = np.minimum(dist_walls, r_neighbors)
    
    return radii

def objective(vars):
    """
    Objective function to minimize: negative sum of radii.
    """
    centers = vars.reshape(26, 2)
    radii = compute_radii(centers)
    # We want to maximize sum of radii, so minimize negative sum
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for centers: [0, 1] for each coordinate
    bounds = [(0.0, 1.0)] * (n * 2)
    
    # Generate multiple initial configurations to search for global optimum
    initial_configs = []
    
    # Config 1: Random uniform
    np.random.seed(42)
    c1 = np.random.rand(n, 2)
    initial_configs.append(c1)
    
    # Config 2: Grid 5x5 plus one extra point
    # 5x5 grid with spacing 0.2 starting at 0.1
    grid = []
    for r in range(5):
        for c in range(5):
            grid.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    # grid has 25 points. Add 26th point.
    # A good spot might be center of a gap or just slightly perturbed
    # Let's add at (0.5, 0.5) is already there (center).
    # Add at (0.05, 0.05) ? Too close to wall.
    # Add at (0.15, 0.15) ?
    grid.append([0.15, 0.15])
    initial_configs.append(np.array(grid))
    
    # Config 3: Perturbed Grid
    c3 = np.array(grid) + np.random.normal(0, 0.01, size=(26, 2))
    c3 = np.clip(c3, 0.01, 0.99)
    initial_configs.append(c3)

    # Config 4: Hexagonal-like packing attempt
    # Rows of circles
    hex_pts = []
    # Approximate radius 0.1, diameter 0.2
    # Vertical spacing 0.1732 (sqrt(3)/2 * 0.2)
    # Horizontal spacing 0.2
    # Let's just place points
    y_start = 0.1
    x_start_even = 0.1
    x_start_odd = 0.2
    dy = 0.1732
    
    y = y_start
    row = 0
    while y <= 0.9:
        x = x_start_even if row % 2 == 0 else x_start_odd
        while x <= 0.9:
            if len(hex_pts) < 26:
                hex_pts.append([x, y])
            x += 0.2
        y += dy
        row += 1
    
    # If we don't have 26, fill with random
    while len(hex_pts) < 26:
        hex_pts.append(np.random.rand(2) * 0.8 + 0.1)
    
    initial_configs.append(np.array(hex_pts[:26]))

    # Optimization loop
    for i, init_c in enumerate(initial_configs):
        # Perturb slightly to avoid exact symmetry issues
        init_perturbed = init_c + np.random.normal(0, 0.005, init_c.shape)
        init_perturbed = np.clip(init_perturbed, 0.001, 0.999)
        
        x0 = init_perturbed.flatten()
        
        try:
            # Use Powell method for derivative-free optimization
            # It works well for non-smooth objective functions
            res = minimize(objective, x0, method='Powell', 
                           bounds=bounds, 
                           options={'maxiter': 3000, 'ftol': 1e-12, 'xtol': 1e-12})
            
            centers = res.x.reshape(n, 2)
            radii = compute_radii(centers)
            current_sum = np.sum(radii)
            
            # Verify validity manually just in case (though compute_radii guarantees it)
            # But check for any numerical weirdness
            valid = True
            for j in range(n):
                if radii[j] < 0:
                    valid = False
                    break
            
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers
                best_radii = radii
                
        except Exception as e:
            pass

    # If optimization failed to find anything good (unlikely), fallback to grid
    if best_sum < 0:
        centers = np.array(initial_configs[1])
        radii = compute_radii(centers)
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii

    return best_centers, best_radii, float(best_sum)

# Note: The problem statement asks for a function run_packing.
# The code above defines it.
