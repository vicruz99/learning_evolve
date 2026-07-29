# sol_000241 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e7c70ed6) state=332c2705 sum of radii=1.097131 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist, squareform

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We start with a hexagonal pattern to leverage high packing density.
    # An initial radius of 0.085 allows fitting circles comfortably in the unit square.
    r_init = 0.085
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    idx = 0
    y = r_init
    row_shift = 0.0
    
    # Generate hex grid points
    # We loop to fill rows, alternating the horizontal shift to create a staggered grid.
    max_rows = 10 
    rows_generated = 0
    while idx < n and rows_generated < max_rows:
        x = r_init + row_shift
        # Fit as many circles as possible in this row
        while x + r_init <= 1 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_init
        
        # Move to next row
        y += math.sqrt(3) * r_init
        rows_generated += 1
        
        # Toggle shift for hex pattern: 0 -> r -> 0 ...
        if row_shift == 0.0:
            row_shift = r_init
        else:
            row_shift = 0.0
            
        if y + r_init > 1:
            break
            
    # Fill remaining spots if any (unlikely with this density)
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    # 2. Optimization using Penalty Method
    # We optimize centers and radii jointly.
    # Variables: [x1, y1, r1, ..., x26, y26, r26]
    
    def objective(vars):
        centers_opt = vars[:2*n].reshape((n, 2))
        radii_opt = vars[2*n:]
        
        # Objective: Maximize sum of radii (so we minimize negative sum)
        obj = -np.sum(radii_opt)
        
        penalty = 0.0
        
        # Boundary penalties: Ensure circles stay within [0, 1]
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        penalty += 1000 * np.sum(np.maximum(0, radii_opt - centers_opt[:, 0])**2)
        penalty += 1000 * np.sum(np.maximum(0, radii_opt - (1 - centers_opt[:, 0]))**2)
        penalty += 1000 * np.sum(np.maximum(0, radii_opt - centers_opt[:, 1])**2)
        penalty += 1000 * np.sum(np.maximum(0, radii_opt - (1 - centers_opt[:, 1]))**2)
        
        # Overlap penalties: Ensure dist(i, j) >= r_i + r_j
        dists = pdist(centers_opt)
        dists_mat = squareform(dists)
        r_sum = radii_opt[:, None] + radii_opt[None, :]
        
        overlaps = r_sum - dists_mat
        penalty += 1000 * np.sum(np.maximum(0, overlaps)**2)
        
        return obj + penalty

    # Define bounds for variables
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x in [0, 1]
        bounds.append((0, 1)) # y in [0, 1]
        bounds.append((0, 0.5)) # r in [0, 0.5]
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12})
        final_vars = res.x
    except Exception:
        final_vars = x0
        
    final_centers = final_vars[:2*n].reshape((n, 2))
    final_radii = final_vars[2*n:]
    
    # 3. Refine radii using Linear Programming (LP)
    # For the optimized centers, we find the exact maximum radii satisfying all constraints.
    n_vars = n
    c = -np.ones(n_vars) # Maximize sum(r) <=> Minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= dist_to_boundary
    for i in range(n):
        x, y = final_centers[i]
        limit = min(x, 1-x, y, 1-y)
        if limit < 0: limit = 0 
        row = np.zeros(n_vars)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(limit)
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dists_mat = squareform(pdist(final_centers))
    for i in range(n):
        for j in range(i + 1, n):
            d = dists_mat[i, j]
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    r_bounds = [(0, None)] * n
    
    # Solve LP
    lp_res = None
    try:
        # Try HiGHS solver (faster/more robust if available)
        lp_res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=r_bounds, method='highs')
    except Exception:
        try:
            # Fallback to default method
            lp_res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=r_bounds)
        except Exception:
            pass
            
    if lp_res and lp_res.success:
        final_radii = lp_res.x
    else:
        # Fallback: if LP fails, ensure radii are valid by shrinking slightly
        dists_mat = squareform(pdist(final_centers))
        r_sum = final_radii[:, None] + final_radii[None, :]
        overlaps = r_sum - dists_mat
        if np.any(overlaps > 1e-9):
            for _ in range(100):
                r_sum = final_radii[:, None] + final_radii[None, :]
                max_ov = np.max(np.maximum(0, r_sum - dists_mat))
                if max_ov < 1e-9: break
                final_radii *= 0.95

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
