# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f64c520b) state=092fb325 sum of radii=2.541421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def get_boundary_limits(centers):
    """
    Calculates the maximum possible radius for each circle based on 
    its distance to the boundaries of the unit square [0,1]x[0,1].
    """
    x = centers[:, 0]
    y = centers[:, 1]
    # Distance to nearest boundary: min(x, 1-x, y, 1-y)
    limits = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    return np.maximum(limits, 0.0)

def solve_radii_lp(centers):
    """
    Solves the Linear Programming problem to find the maximum sum of radii
    for a fixed set of centers.
    Maximize sum(r_i)
    Subject to:
        r_i >= 0
        r_i <= boundary_limit_i
        r_i + r_j <= distance(i, j) for all i < j
    """
    n = centers.shape[0]
    limits = get_boundary_limits(centers)
    
    # Objective: minimize -sum(r)
    c = np.ones(n) * -1
    
    # Bounds: 0 <= r_i <= limits[i]
    bounds = [(0, lim) for lim in limits]
    
    # Precompute distances between all pairs of centers
    # diff shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Construct constraints matrix A_ub and vector b_ub
    # Constraints: r_i + r_j <= dists[i, j]
    n_constraints = n * (n - 1) // 2
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        # Try to solve LP using 'highs' method (fast), fallback to 'simplex'
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except ValueError:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
            
        if res.success:
            return res.x
        else:
            # Fallback: if LP fails, use a safe heuristic
            # Radius is limited by half the distance to nearest neighbor and boundaries
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            safe_r = np.minimum(limits, min_dists / 2.0)
            return np.maximum(safe_r, 0.0)
    except:
        # Fallback in case of any error
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        safe_r = np.minimum(limits, min_dists / 2.0)
        return np.maximum(safe_r, 0.0)

def score(centers_flat):
    """
    Objective function for optimization.
    Returns negative sum of radii (since minimize seeks to minimize).
    """
    centers = centers_flat.reshape(-1, 2)
    radii = solve_radii_lp(centers)
    return -np.sum(radii)

def run_packing():
    """
    Runs the optimization to pack 26 circles in a unit square.
    """
    n = 26
    best_score = np.inf
    best_centers = None
    
    candidates = []
    
    # Initialization 1: 5x5 Grid + 1 circle in the center gap
    # A 5x5 grid with spacing 0.2 fits 25 circles of radius 0.1.
    # We add a 26th circle in the center of a hole.
    coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    points = []
    for x in coords:
        for y in coords:
            points.append([x, y])
    # Add extra circle at (0.2, 0.2) which is in the gap between (0.1,0.1), (0.1,0.3), etc.
    points.append([0.2, 0.2])
    candidates.append(np.array(points))
    
    # Initialization 2-5: Random configurations to escape local minima
    np.random.seed(42)
    for _ in range(4):
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        candidates.append(centers)
        
    # Optimize each candidate
    for cand in candidates:
        x0 = cand.flatten()
        try:
            # Use Nelder-Mead to optimize centers
            res = minimize(score, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_centers = res.x.reshape(-1, 2)
        except:
            pass
            
    if best_centers is None:
        best_centers = candidates[0]
        
    # Ensure centers are strictly within [0, 1]
    best_centers = np.clip(best_centers, 0, 1)
    
    # Compute final optimal radii for the best centers found
    final_radii = solve_radii_lp(best_centers)
    final_radii = np.maximum(final_radii, 0.0)
    
    return best_centers, final_radii, float(np.sum(final_radii))
