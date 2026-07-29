# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e9cb3956) state=41e86c65 sum of radii=0.864531 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Generate a compact hexagonal cluster of points
    # We generate points on a hexagonal lattice and select the 26 closest to the origin
    # to ensure a compact shape that fits well in a square.
    candidates = []
    # Sufficient range to cover 26 points comfortably
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i + j * 0.5
            y = j * np.sqrt(3) / 2
            candidates.append((x, y))
            
    # Sort by distance from origin to pick the most compact cluster
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_points = candidates[:n]
    centers = np.array(selected_points)
    
    # 2. Scale and center the points in [0, 1] x [0, 1]
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    width = max_c[0] - min_c[0]
    height = max_c[1] - min_c[1]
    
    # Scale to fit with some margin to allow for optimization movement
    scale = 0.9 / max(width, height)
    centers = (centers - min_c) * scale
    centers = centers + 0.5 - (centers.max(axis=0) - centers.min(axis=0)) / 2
    
    # 3. Force-directed optimization to expand radii
    # We simulate growing circles and repelling them to fit in the square.
    # This helps find a good configuration of centers.
    radii = np.full(n, 0.05) # Initial radius guess
    lr_pos = 0.0005
    lr_rad = 0.0001
    
    # Run simulation for a number of steps
    for step in range(3000):
        # Increase radii slightly
        radii += lr_rad
        
        force = np.zeros_like(centers)
        
        # Boundary repulsion: push circles away from walls if they exceed radius
        for i in range(n):
            x, y = centers[i]
            r_i = radii[i]
            
            if x < r_i:
                force[i, 0] += (r_i - x) * 100
            if x > 1 - r_i:
                force[i, 0] -= (x - (1 - r_i)) * 100
            if y < r_i:
                force[i, 1] += (r_i - y) * 100
            if y > 1 - r_i:
                force[i, 1] -= (y - (1 - r_i)) * 100
        
        # Neighbor repulsion: push overlapping circles apart
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Repulsive force proportional to overlap
                    fx = (dx / dist) * overlap * 100
                    fy = (dy / dist) * overlap * 100
                    force[i, 0] += fx
                    force[i, 1] += fy
                    force[j, 0] -= fx
                    force[j, 1] -= fy
        
        # Update positions
        centers += force * lr_pos
        centers = np.clip(centers, 0.0, 1.0)
    
    # 4. Solve Linear Programming problem to maximize sum of radii
    # for the fixed optimal positions found above.
    # Maximize sum(r_i)
    # subject to r_i + r_j <= dist(i,j) and r_i <= dist(i, boundary)
    
    # Precompute distances and boundary limits
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
    boundary_limits = np.array([
        min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for i in range(n)
    ])
    
    # LP formulation
    c = -np.ones(n) # Maximize sum(r) -> Minimize -sum(r)
    A_ub = []
    b_ub = []
    
    # Constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Constraints: r_i <= boundary_limit(i)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(boundary_limits[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        # Use 'highs' method which is robust and fast
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            final_radii = res.x
            final_sum = -res.fun
            return centers, final_radii, final_sum
    except Exception:
        pass
        
    # Fallback to equal radii if LP fails
    min_r = 1.0
    for i in range(n):
        lim = boundary_limits[i]
        for j in range(n):
            if i != j:
                lim = min(lim, dists[i, j] / 2)
        if lim < min_r:
            min_r = lim
            
    final_radii = np.full(n, max(min_r, 0))
    return centers, final_radii, np.sum(final_radii)
