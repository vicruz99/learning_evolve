import numpy as np
from scipy.optimize import differential_evolution

def get_radii_from_centers(centers):
    """
    Computes the maximum possible radii for circles centered at 'centers'
    such that they do not overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.full(n, 1.0) # Initialize with large value

    # 1. Boundary constraints: r <= min(x, 1-x, y, 1-y)
    x = centers[:, 0]
    y = centers[:, 1]
    dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # 2. Neighbor constraints: r_i + r_j <= dist(i, j)
    # This is a system of inequalities. For a fixed set of centers, 
    # the optimal radii for maximizing sum of radii are generally when 
    # r_i = r_j = dist(i,j)/2 for the closest pair.
    # However, to strictly satisfy r_i + r_j <= dist(i,j), 
    # a simple conservative estimate is r_i = 0.5 * min_j(dist(i,j)).
    # This guarantees validity. 
    # (Note: For unequal circles, a more complex system solves for exact radii, 
    # but the symmetric approximation is a very strong lower bound and 
    # usually near-optimal for maximizing sum).
    
    for i in range(n):
        min_dist = dist_boundary[i]
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist / 2.0 < min_dist:
                    min_dist = dist / 2.0
        radii[i] = min_dist

    return radii

def objective(centers_flat):
    """
    Objective function to minimize (negative sum of radii).
    """
    n = 26
    centers = centers_flat.reshape(-1, 2)
    
    # Clip to ensure centers are within [0, 1]
    # (Though differential_evolution bounds should handle this)
    centers = np.clip(centers, 0, 1)
    
    radii = get_radii_from_centers(centers)
    return -np.sum(radii)

def create_hex_grid(n):
    """
    Generates a hexagonal grid of n points in [0,1]x[0,1] as a good initial guess.
    """
    points = []
    # Estimate grid size to fit n points
    # Area of hex cell ~ sqrt(3)/2 * s^2. 1 cell per point.
    # s ~ sqrt(2/(sqrt(3)*n))
    s = np.sqrt(2 / (np.sqrt(3) * n))
    
    y = 0
    while y <= 1:
        x = 0
        row_offset = (int(y / (s * np.sqrt(3)/2)) % 2) * (s / 2)
        while x <= 1:
            if len(points) < n:
                # Center the grid roughly in the square
                # Scale and shift coordinates
                # A simpler approach: just generate points and scale them to fit [0,1]
                points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        
    if len(points) == 0: return np.random.rand(n, 2)
    
    pts = np.array(points[:n])
    
    # Normalize to [0, 1] with some padding
    min_c = np.min(pts, axis=0)
    max_c = np.max(pts, axis=0)
    scale = np.array([1, 1]) - 2*0.05 # 10% padding
    pts = (pts - min_c) / (max_c - min_c) * scale + 0.05
    
    return pts

def run_packing() -> tuple:
    n = 26
    bounds = [(0, 1)] * (2 * n)
    
    # Strategy: Use Differential Evolution for robust global optimization
    # It is slower but less likely to get stuck in bad local minima than Nelder-Mead
    # We use a hex grid as a seed to guide the population if possible, 
    # but DE handles its own population.
    
    # Initial seed population for DE to bias towards good structures
    # DE uses 'init' parameter. 
    # We can't easily pass a full population, so we rely on random init 
    # but perhaps run a local search from a hex grid too.
    
    # Let's run DE
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1500, 
        popsize=30, 
        mutation=(0.5, 1.5), 
        recombination=0.9,
        tol=1e-7,
        polish=True
    )
    
    best_centers = res.x.reshape(n, 2)
    best_radii = get_radii_from_centers(best_centers)
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum