# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05a03f22) state=d072e8a6 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates a valid packing of 26 circles in a unit square [0,1]x[0,1]
    maximizing the sum of radii.
    """
    np.random.seed(42)
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initialization: Hexagonal lattice
    s = 0.18  # spacing between lattice points
    points = []
    y = 0.1
    row_idx = 0
    # Generate points covering the square
    while y <= 0.95:
        if row_idx % 2 == 0:
            start_x = 0.1
            step_x = s
        else:
            start_x = 0.1 + s/2
            step_x = s
        
        x = start_x
        while x <= 0.95:
            points.append([x, y])
            x += step_x
        
        y += s * math.sqrt(3) / 2
        row_idx += 1
    
    # Fallback grid if lattice generation is insufficient
    if len(points) < n:
        k = int(math.ceil(math.sqrt(n)))
        for i in range(k):
            for j in range(k):
                if i*k + j < n:
                    points.append([0.1 + i*0.8/(k-1), 0.1 + j*0.8/(k-1)])
        points = points[:n]

    # Select n points closest to center to distribute circles well
    dists = [math.sqrt((p[0]-0.5)**2 + (p[1]-0.5)**2) for p in points]
    sorted_indices = np.argsort(dists)
    selected = [points[i] for i in sorted_indices[:n]]
    centers = np.array(selected)
    
    # Initial radii - small enough to avoid immediate overlap
    radii[:] = 0.05
    
    # Optimization Loop: Expand radii and fix positions
    for iteration in range(300):
        # 1. Position Optimization (Repulsion)
        for _ in range(50):
            moved = False
            order = np.random.permutation(n)
            for i in order:
                for j in range(n):
                    if i == j: continue
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9: 
                        dist = 1e-9
                        dx, dy = 1e-9, 0.0 
                    
                    req = radii[i] + radii[j]
                    if dist < req:
                        # Push apart
                        vec_x = dx / dist
                        vec_y = dy / dist
                        push = (req - dist) * 0.5
                        centers[i, 0] += vec_x * push
                        centers[i, 1] += vec_y * push
                        centers[j, 0] -= vec_x * push
                        centers[j, 1] -= vec_y * push
                        moved = True
            
            # Boundary Constraints
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x < r: x = r
                elif x > 1 - r: x = 1 - r
                if y < r: y = r
                elif y > 1 - r: y = 1 - r
                if x != centers[i, 0] or y != centers[i, 1]:
                    centers[i, 0] = x
                    centers[i, 1] = y
                    moved = True
            if not moved: break
        
        # 2. Expand Radii
        max_exp = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Distance to boundaries
            lim = min(x - r, 1 - (x + r), y - r, 1 - (y + r))
            if lim < 0: lim = 0
            
            # Distance to neighbors
            for j in range(n):
                if i == j: continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                gap = dist - radii[j] - radii[i]
                if gap < 0: gap = 0
                lim = min(lim, gap / 2.0)
            max_exp[i] = lim
        
        # Apply uniform growth
        min_exp = np.min(max_exp)
        if min_exp > 1e-7:
            radii += min_exp
        else:
            # Perturb constrained circles
            constrained = np.where(max_exp < 1e-4)[0]
            if len(constrained) > 0:
                for i in constrained:
                    centers[i] += np.random.normal(0, 0.005, 2)
                    r = radii[i]
                    centers[i, 0] = max(r, min(1-r, centers[i, 0]))
                    centers[i, 1] = max(r, min(1-r, centers[i, 1]))

    # Final strict validation and repair
    for _ in range(200):
        overlap = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j]:
                    if dist < 1e-9:
                        dx, dy = 0.01, 0.01
                        dist = math.sqrt(dx*dx + dy*dy)
                    vec_x = dx / dist
                    vec_y = dy / dist
                    push = (radii[i] + radii[j] - dist) * 0.5 + 1e-5
                    centers[i, 0] += vec_x * push
                    centers[i, 1] += vec_y * push
                    centers[j, 0] -= vec_x * push
                    centers[j, 1] -= vec_y * push
                    overlap = True
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            centers[i, 0] = x
            centers[i, 1] = y
            
        if not overlap: break

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
