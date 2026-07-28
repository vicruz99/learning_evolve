# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state 6f2d6856) state=ec408f22 sum of radii=2.351462 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_smooth_min_clearance(centers, k):
    """
    Computes a smooth approximation of the minimum clearance to boundaries and other circles.
    """
    n = centers.shape[0]
    # Boundary clearances: distance to nearest wall
    b_clr = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                       np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise clearances: half the distance between circle centers
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    p_clr = dists / 2.0
    
    # Combine all constraints into one array, excluding diagonal infinities
    all_clr = np.concatenate([b_clr, p_clr[p_clr < np.inf]])
    
    # Smooth approximation of the minimum using log-sum-exp trick for numerical stability
    m = np.max(all_clr)
    return m - np.log(np.sum(np.exp(-k * (all_clr - m)))) / k

def objective(centers_flat, n, k):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes negative smooth minimum clearance (equivalent to maximizing it).
    """
    centers = centers_flat.reshape(n, 2)
    return -compute_smooth_min_clearance(centers, k)

def solve_optimal_radii(centers):
    """
    Solves a Linear Programming problem to find radii that maximize sum(r_i)
    subject to non-overlap and boundary constraints for fixed centers.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    # Maximum radius allowed by boundaries for each circle
    wall_limits = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # LP setup: Maximize sum(r) <=> Minimize -sum(r)
    c = -np.ones(n)
    bounds = [(0.0, lim) for lim in wall_limits]
    
    # Pairwise distance constraints: r_i + r_j <= dist(i, j)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        # 'highs' is a fast, robust LP solver available in modern scipy
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    # Fallback: equal radii based on strict minimum clearance
    min_d = np.min(np.concatenate([wall_limits, dists/2.0]))
    return np.full(n, max(0.0, min_d - 1e-7))

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Strategy:
    1. Initialize with a hexagonal lattice pattern.
    2. Optimize center positions to maximize the minimum clearance using a smooth approximation.
    3. Solve an LP to find the exact maximum sum of radii for the optimized centers.
    """
    n = 26
    k = 30.0  # Sharpness parameter for the smooth min approximation
    
    # 1. Initialize with a dense hexagonal lattice
    r_init = 0.1
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    y = r_init
    dy = np.sqrt(3) * r_init
    dx = 2 * r_init
    
    while idx < n:
        shift = dx / 2.0 if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and idx < n:
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            x += dx
        y += dy
        row += 1
        
    # Normalize to fit comfortably within [0, 1]^2 while maintaining relative structure
    centers -= centers.min(axis=0)
    centers /= centers.max(axis=0)
    centers *= 0.9
    centers += 0.05
    
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    best_val = np.inf
    best_centers = centers
    
    np.random.seed(42)
    # 2. Optimize center positions with multiple restarts to escape local minima
    for _ in range(8):
        curr = best_centers + np.random.uniform(-0.04, 0.04, size=(n, 2))
        curr = np.clip(curr, 0.01, 0.99)
        
        res = minimize(objective, curr.flatten(), args=(n, k), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 15000, 'ftol': 1e-13})
        
        if res.fun < best_val:
            best_val = res.fun
            best_centers = res.x.reshape(n, 2)
            
    # 3. Solve LP to find the exact maximum sum of radii for the optimized centers
    radii = solve_optimal_radii(best_centers)
    
    # Apply a tiny safety margin to satisfy strict numerical validation tolerances
    radii *= 0.999995
    
    return best_centers, radii, np.sum(radii)
