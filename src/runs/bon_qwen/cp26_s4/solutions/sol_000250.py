# sol_000250 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 30e75f73) state=04dc7d49 sum of radii=0.716018 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses an iterative growing circles approach with relaxation.
    """
    n = 26
    # Initialize centers in a pattern: 5 rows with 5, 5, 6, 5, 5 circles
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.02
    
    idx = 0
    row_counts = [5, 5, 6, 5, 5]
    y_pos = 0.15 
    
    for count in row_counts:
        # Distribute x evenly in [0.05, 0.95]
        x_coords = np.linspace(0.05, 0.95, count)
        for x in x_coords:
            centers[idx] = [x, y_pos]
            idx += 1
        y_pos += 0.2
    
    # Optimization loop
    max_iter = 5000
    alpha_expand = 0.1
    alpha_relax = 0.5
    
    for iteration in range(max_iter):
        slacks = np.full(n, 1.0)
        
        for i in range(n):
            x, y = centers[i]
            r_i = radii[i]
            
            # Wall slacks
            slack = min(x, 1 - x, y, 1 - y) - r_i
            
            # Neighbor slacks
            for j in range(n):
                if i == j: continue
                xj, yj = centers[j]
                r_j = radii[j]
                dist = math.hypot(x - xj, y - yj)
                space = dist - r_j
                if space < slack:
                    slack = space
            
            slacks[i] = slack
        
        # Expand
        for i in range(n):
            if slacks[i] > 1e-7:
                radii[i] += alpha_expand * slacks[i]
        
        radii = np.minimum(radii, 0.5)
        
        # Relaxation
        forces = np.zeros((n, 2))
        
        for i in range(n):
            x, y = centers[i]
            r_i = radii[i]
            
            # Wall forces
            if x < r_i:
                forces[i, 0] += (r_i - x)
            elif x > 1 - r_i:
                forces[i, 0] -= (x - (1 - r_i))
                
            if y < r_i:
                forces[i, 1] += (r_i - y)
            elif y > 1 - r_i:
                forces[i, 1] -= (y - (1 - r_i))
            
            for j in range(i + 1, n):
                xj, yj = centers[j]
                r_j = radii[j]
                dx = x - xj
                dy = y - yj
                dist = math.hypot(dx, dy)
                
                min_dist = r_i + r_j
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                    else:
                        fx = np.random.rand() - 0.5
                        fy = np.random.rand() - 0.5
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        centers += forces * alpha_relax
        centers = np.clip(centers, 0, 1)

    # Repair step to ensure strict validity
    # 1. Ensure circles are inside square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1 - x, y, 1 - y)
        if r > max_r:
            radii[i] = max_r
            
    # 2. Resolve any remaining overlaps by shrinking radii
    for _ in range(100):
        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r - 1e-12:
                    has_overlap = True
                    overlap = sum_r - dist
                    if radii[i] + radii[j] > 1e-9:
                        factor_i = radii[i] / (radii[i] + radii[j])
                        factor_j = radii[j] / (radii[i] + radii[j])
                        radii[i] -= 0.5 * overlap * factor_i
                        radii[j] -= 0.5 * overlap * factor_j
                    else:
                        radii[i] = 0
                        radii[j] = 0
        if not has_overlap:
            break
            
    radii = np.maximum(radii, 0)
    
    return centers, radii, float(np.sum(radii))
