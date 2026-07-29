# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6d8d18a8) state=ad542d5c sum of radii=2.430602 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """
    Solves for the radii that maximize sum of radii for fixed centers using Linear Programming.
    
    Problem:
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(center_i, center_j) for all i < j
      0 <= r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
    """
    n = centers.shape[0]
    # Objective: Maximize sum(r_i) <=> Minimize -sum(r_i)
    c = -np.ones(n)
    bounds = []
    A_ub = []
    b_ub = []
    
    # Calculate boundary distances and inter-circle distances
    for i in range(n):
        x, y = centers[i]
        # Distance to nearest boundary
        r_max = min(x, 1-x, y, 1-y)
        if r_max < 0:
            r_max = 0.0
        bounds.append((0, r_max))
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            return None, 0.0
    except Exception:
        return None, 0.0

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Initialization: Hexagonal lattice pattern
    # We construct a pattern that fits 26 circles in a roughly hexagonal arrangement.
    # Rows configuration: 5, 5, 5, 5, 4, 2 (Total 26)
    points = []
    sqrt3 = math.sqrt(3)
    
    # Row 0: 5 circles
    for i in range(5): points.append((i * 2.0, 0.0))
    # Row 1: 5 circles (shifted by 1 unit in x)
    for i in range(5): points.append((i * 2.0 + 1.0, sqrt3))
    # Row 2: 5 circles
    for i in range(5): points.append((i * 2.0, 2.0 * sqrt3))
    # Row 3: 5 circles (shifted)
    for i in range(5): points.append((i * 2.0 + 1.0, 3.0 * sqrt3))
    # Row 4: 4 circles
    for i in range(4): points.append((i * 2.0, 4.0 * sqrt3))
    # Row 5: 2 circles (shifted)
    for i in range(2): points.append((i * 2.0 + 1.0, 5.0 * sqrt3))
    
    # Normalize coordinates to fit in [0, 1]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    
    centers = np.zeros((n, 2))
    for i, (x, y) in enumerate(points):
        # Add small random noise to break symmetry and avoid local minima
        nx = (x - min_x) / w + np.random.normal(0, 0.005)
        ny = (y - min_y) / h + np.random.normal(0, 0.005)
        # Clip to keep inside unit square with margin
        centers[i] = [np.clip(nx, 0.05, 0.95), np.clip(ny, 0.05, 0.95)]
        
    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = np.ones(n)
    
    # Phase 1: Coarse local search to find a good region
    temp = 0.05
    for it in range(1000):
        idx = random.randint(0, n-1)
        old_pos = centers[idx].copy()
        
        # Perturb position
        dx = np.random.normal(0, temp)
        dy = np.random.normal(0, temp)
        new_x = np.clip(old_pos[0] + dx, 0.01, 0.99)
        new_y = np.clip(old_pos[1] + dy, 0.01, 0.99)
        centers[idx] = [new_x, new_y]
        
        # Evaluate optimal radii for new positions
        radii, current_sum = get_optimal_radii(centers)
        
        if radii is not None and current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            # Slight cooling
            temp *= 0.998 
        else:
            centers[idx] = old_pos
            temp *= 0.999
            
    # Phase 2: Fine local search to polish the solution
    centers = best_centers.copy()
    temp = 0.005
    for it in range(1000):
        idx = random.randint(0, n-1)
        old_pos = centers[idx].copy()
        
        dx = np.random.normal(0, temp)
        dy = np.random.normal(0, temp)
        new_x = np.clip(old_pos[0] + dx, 0.01, 0.99)
        new_y = np.clip(old_pos[1] + dy, 0.01, 0.99)
        centers[idx] = [new_x, new_y]
        
        radii, current_sum = get_optimal_radii(centers)
        
        if radii is not None and current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
        else:
            centers[idx] = old_pos
            
    return best_centers, best_radii, best_sum
