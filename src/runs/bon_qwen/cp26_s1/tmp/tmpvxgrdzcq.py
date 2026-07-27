import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))

    # 1. Initial placement in a hexagonal/staggered grid
    # Arrangement: 6-5-6-5-4 circles in rows to fit 26
    row_counts = [6, 5, 6, 5, 4]
    y_step = 1.0 / 5.2 
    x_step = 1.0 / 7.0
    y_start = 0.15
    idx = 0
    for i, count in enumerate(row_counts):
        x_start = (0.0 if i % 2 == 0 else x_step)
        for j in range(count):
            centers[idx, 0] = x_start + j * 2 * x_step
            centers[idx, 1] = y_start + i * y_step
            idx += 1

    # 2. Optimization function to maximize minimum distance
    def objective(vars_flat):
        c = vars_flat.reshape(-1, 2)
        min_dist = float('inf')
        # Distance to boundaries (scaled by 2 to match pairwise logic)
        for i in range(n):
            min_dist = min(min_dist, c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        
        # Pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                min_dist = min(min_dist, dist)
        
        return -min_dist

    res = minimize(objective, centers.flatten(), method='SLSQP')
    centers_opt = res.x.reshape(-1, 2)

    # 3. Determine radii based on the optimized clearance
    min_dist = float('inf')
    for i in range(n):
        min_dist = min(min_dist, centers_opt[i, 0], 1.0 - centers_opt[i, 0],
                       centers_opt[i, 1], 1.0 - centers_opt[i, 1])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j]) ** 2))
            min_dist = min(min_dist, dist)
    
    radii = np.full(n, min_dist / 2.0)
    total_radii = np.sum(radii)
    
    return centers_opt, radii, total_radii