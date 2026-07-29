# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9227c4d6) state=b447a8b4 sum of radii=2.387893 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def solve_radii_lp(centers, n):
    """
    Given centers of n circles, find optimal radii to maximize sum of radii
    subject to non-overlap and boundary constraints using Linear Programming.
    
    Args:
        centers: np.array of shape (n, 2)
        n: int, number of circles
        
    Returns:
        np.array of shape (n) with optimal radii
    """
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints:
    # 1. r_i + r_j <= dist(i, j) for all i < j
    # 2. r_i <= dist(i, wall) for all i
    
    # Compute pairwise distances
    # centers shape (n, 2)
    # Using broadcasting to compute distance matrix
    # diff_x[i, j] = centers[i, 0] - centers[j, 0]
    diff_x = centers[:, 0:1] - centers[:, 0:1].T
    diff_y = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(diff_x**2 + diff_y**2)
    
    # Number of constraints
    num_pairs = n * (n - 1) // 2
    num_walls = n
    num_constraints = num_pairs + num_walls
    
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    row = 0
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = dists[i, j]
            row += 1
            
    # Wall constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        r_max = min(x, 1.0 - x, y, 1.0 - y)
        A_ub[row, i] = 1.0
        b_ub[row] = r_max
        row += 1
        
    bounds = [(0.0, 1.0)] * n
    
    try:
        # 'highs' is a fast LP solver available in recent SciPy versions
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    # Fallback greedy assignment if LP fails
    # r_i = min(wall_dist, 0.5 * min_neighbor_dist)
    radii = np.zeros(n)
    for i in range(n):
        # Find min distance to other centers (ignore self distance 0)
        min_d = np.inf
        for j in range(n):
            if i != j and dists[i, j] < min_d:
                min_d = dists[i, j]
        
        r_wall = min(centers[i, 0], 1.0-centers[i, 0], centers[i, 1], 1.0-centers[i, 1])
        radii[i] = min(r_wall, 0.5 * min_d)
    return radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # 1. Initial Configuration: Hexagonal Packing
    # We generate points on a hexagonal grid and select n points closest to center.
    # This provides a dense, valid starting point.
    
    r_est = 0.09
    w = 2 * r_est
    h = np.sqrt(3) * r_est
    
    points = []
    y = r_est
    row_idx = 0
    # Generate enough points
    while y + r_est <= 1.0:
        shift = r_est if (row_idx % 2 == 1) else 0.0
        x = r_est + shift
        while x + r_est <= 1.0:
            points.append([x, y])
            x += w
        y += h
        row_idx += 1
        
    points = np.array(points)
    
    # Select n points
    if len(points) >= n:
        # Distance to center (0.5, 0.5)
        dists2 = np.sum((points - 0.5)**2, axis=1)
        idx = np.argsort(dists2)[:n]
        centers_init = points[idx]
    else:
        centers_init = points
        # If not enough, add random points
        while len(centers_init) < n:
            pt = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
            centers_init = np.vstack([centers_init, pt])
        centers_init = centers_init[:n]
        
    # 2. Optimization
    # We maximize sum of radii by optimizing centers.
    # Radii are determined by LP for each center configuration.
    
    def objective(centers_flat):
        centers = centers_flat.reshape((n, 2))
        centers = np.clip(centers, 0.0, 1.0)
        radii = solve_radii_lp(centers, n)
        return -np.sum(radii)
        
    x0 = centers_init.flatten()
    
    # Run optimization
    # Nelder-Mead is robust for non-smooth objective (LP solution)
    # 2000 iterations is sufficient to refine the hexagonal layout
    res = opt.minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
    
    best_centers = np.clip(res.x.reshape((n, 2)), 0.0, 1.0)
    best_radii = solve_radii_lp(best_centers, n)
    sum_radii = float(np.sum(best_radii))
    
    return best_centers, best_radii, sum_radii
