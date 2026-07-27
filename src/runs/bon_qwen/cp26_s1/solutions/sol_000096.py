# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5526c41b) state=e31537af sum of radii=1.905501 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """
    Solves the LP to find maximum sum of radii for fixed centers.
    Maximizes sum(r_i) subject to r_i + r_j <= ||c_i - c_j|| and boundary constraints.
    """
    n = centers.shape[0]
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Objective: maximize sum(r) <=> minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Constraints: r_i + r_j <= dists[i,j] for all i < j
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    # Bounds: 0 <= r_i <= distance to nearest boundary
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max_r))
        
    # Solve LP
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return np.zeros(n), 0.0

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initialization: Hexagonal pattern for efficient packing baseline
    pts = []
    rows = [6, 5, 6, 5, 4]  # Sums to 26
    y = 0.0
    for idx, cnt in enumerate(rows):
        x_start = 0.5 if idx % 2 == 1 else 0.0
        for k in range(cnt):
            pts.append([x_start + k, y])
        y += np.sqrt(3) / 2.0
        
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    centers = np.zeros((n, 2))
    for i, p in enumerate(pts):
        # Scale to [0.15, 0.85] with small jitter
        cx = 0.15 + 0.7 * (p[0] - min_x) / (max_x - min_x)
        cy = 0.15 + 0.7 * (p[1] - min_y) / (max_y - min_y)
        centers[i] = [cx + np.random.normal(0, 0.005), cy + np.random.normal(0, 0.005)]
        
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    step_size = 0.045
    for iter_num in range(1000):
        radii, s = get_optimal_radii(centers)
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute repulsion forces based on tight constraints
        forces = np.zeros_like(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Pairwise repulsion when circles touch/overlap
        for i in range(n):
            for j in range(i + 1, n):
                required = radii[i] + radii[j]
                actual = dists[i, j]
                if actual < required - 1e-9:
                    overlap = required - actual
                    diff_ij = centers[i] - centers[j]
                    dist = np.linalg.norm(diff_ij)
                    if dist > 1e-9:
                        dir_vec = diff_ij / dist  # Points from j to i
                        push = overlap * 0.5 * step_size
                        forces[i] += dir_vec * push
                        forces[j] -= dir_vec * push
                        
        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            margin = 1e-5
            if x - r < margin: forces[i, 0] += (margin - (x - r)) * step_size
            elif x + r > 1 - margin: forces[i, 0] -= (x + r - (1 - margin)) * step_size
            if y - r < margin: forces[i, 1] += (margin - (y - r)) * step_size
            elif y + r > 1 - margin: forces[i, 1] -= (y + r - (1 - margin)) * step_size
            
        # Apply forces and annealing noise
        noise = np.random.normal(0, step_size, size=centers.shape)
        centers = centers + forces + noise
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        
        # Anneal step size
        step_size *= 0.9985
        
        # Periodic local restart to avoid stagnation
        if iter_num > 0 and iter_num % 250 == 0:
            centers = np.random.rand(n, 2) * 0.6 + 0.2
            step_size = 0.04
            
    # Final safety adjustment to strictly satisfy constraints numerically
    if best_radii is not None:
        best_radii = np.maximum(best_radii * 0.99999, 0.0)
        
    return best_centers, best_radii, best_sum
