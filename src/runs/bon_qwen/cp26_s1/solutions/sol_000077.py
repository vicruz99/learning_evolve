# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ed1177e6) state=776f6efe sum of radii=1.720680 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import linprog

def get_max_radii_lp(centers):
    """
    Given fixed centers, solve LP to maximize sum of radii.
    Constraints: r_i + r_j <= dist(i, j), r_i <= boundaries.
    """
    n = len(centers)
    if n == 0:
        return np.array([])

    # Objective: Maximize sum(r) -> Minimize -sum(r)
    c = -np.ones(n)

    A_ub = []
    b_ub = []

    # Compute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    for i in range(n):
        x, y = centers[i]
        
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(x)
        
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(1.0 - x)
        
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(y)
        
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(1.0 - y)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return np.full(n, 1e-9)
    except Exception:
        return np.full(n, 1e-9)

def jiggling_optimize(centers, iterations=2000):
    """
    Refine centers to spread them out using force-directed method.
    """
    n = len(centers)
    if n == 0:
        return centers
        
    radii = np.full(n, 0.05)
    
    for t in range(iterations):
        # Update radii estimate periodically based on current configuration
        if t % 20 == 0:
            x = centers[:, 0]
            y = centers[:, 1]
            b_dist = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
            
            diff = centers[:, None, :] - centers[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_neighbor_dist = np.min(dists, axis=1)
            
            # Heuristic radius
            radii = np.minimum(b_dist, 0.5 * min_neighbor_dist)
            radii = np.maximum(radii, 1e-5)

        # Calculate repulsion forces
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists_safe = np.where(dists == 0, 1e-9, dists)
        
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(r_sum - dists, 0)
        force_mag = overlap * 5.0
        
        force_vecs = -force_mag[:, :, np.newaxis] * unit_vecs
        forces = np.sum(force_vecs, axis=1)
        
        # Boundary forces
        forces[:, 0] += np.where(centers[:, 0] < radii, (radii - centers[:, 0]) * 10.0, 0.0)
        forces[:, 0] -= np.where(centers[:, 0] > 1.0 - radii, (centers[:, 0] - (1.0 - radii)) * 10.0, 0.0)
        forces[:, 1] += np.where(centers[:, 1] < radii, (radii - centers[:, 1]) * 10.0, 0.0)
        forces[:, 1] -= np.where(centers[:, 1] > 1.0 - radii, (centers[:, 1] - (1.0 - radii)) * 10.0, 0.0)
        
        step = 0.01 * (1.0 - t / iterations)
        centers += forces * step
        centers = np.clip(centers, 0.0, 1.0)
        
    return centers

def run_packing() -> tuple:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Multiple random restarts to find global optimum
    seeds = [42, 123, 456, 789, 1000, 2000, 3000]
    
    for seed in seeds:
        np.random.seed(seed)
        current_centers = np.random.rand(N, 2)
        
        # Jiggling to optimize positions
        current_centers = jiggling_optimize(current_centers, iterations=3000)
        
        # LP to optimize radii
        current_radii = get_max_radii_lp(current_centers)
        current_sum = np.sum(current_radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()

    # Final refinement on the best found configuration
    if best_centers is not None:
        best_centers = jiggling_optimize(best_centers, iterations=5000)
        best_radii = get_max_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    if best_centers is None:
        # Fallback grid packing
        best_centers = np.zeros((N, 2))
        best_radii = np.full(N, 0.1)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N:
                    best_centers[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
                    idx += 1
        best_sum = 2.5
        
    return best_centers, best_radii, float(best_sum)
