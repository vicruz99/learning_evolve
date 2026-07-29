# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=8342d16e sum of radii=2.596124 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and non-overlap."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    cons = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    ci = c[pair_i]
    cj = c[pair_j]
    ri = r[pair_i]
    rj = r[pair_j]
    dist = np.sqrt(np.sum((ci - cj)**2, axis=1))
    cons.append(dist - ri - rj)
    
    return np.concatenate(cons)

def run_packing():
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Hexagonal lattices with various shifts
    for ox in [0.0, 0.04, 0.08]:
        for oy in [0.0, 0.04]:
            pts = []
            y = 0.06 + oy
            row = 0
            while y < 0.94:
                x = 0.06 + ox + (row % 2) * 0.055
                while x < 0.94:
                    pts.append([x, y])
                    x += 0.11
                y += 0.095
                row += 1
            pts = np.array(pts[:n])
            pts += np.random.uniform(-0.005, 0.005, pts.shape)
            pts = np.clip(pts, 0.05, 0.95)
            r_init = np.full(n, 0.05)
            starts.append(np.concatenate([pts.flatten(), r_init]))
            
    # 2. Staggered grids
    for step in [0.13, 0.15, 0.17]:
        pts = []
        x = step / 2
        while x < 1.0:
            y = step / 2
            while y < 1.0:
                pts.append([x, y])
                y += step
            x += step
        pts = np.array(pts[:n])
        pts += np.random.uniform(-0.005, 0.005, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        r_init = np.full(n, 0.05)
        starts.append(np.concatenate([pts.flatten(), r_init]))
        
    # 3. Random valid starts
    np.random.seed(42)
    for _ in range(4):
        pts = np.random.uniform(0.1, 0.9, (n, 2))
        r_init = np.full(n, 0.04)
        starts.append(np.concatenate([pts.flatten(), r_init]))
        
    # Run optimization for each start
    for x0 in starts:
        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pair_i, pair_j)},
                       options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                       
        centers = res.x[:2*n].reshape(n, 2)
        radii = res.x[2*n:]
        
        # Deterministic validity enforcement via uniform scaling
        min_scale = 1.0
        
        # Check boundary constraints
        for i in range(n):
            if radii[i] > 1e-14:
                min_scale = min(min_scale, 
                                centers[i, 0] / radii[i],
                                (1.0 - centers[i, 0]) / radii[i],
                                centers[i, 1] / radii[i],
                                (1.0 - centers[i, 1]) / radii[i])
                                
        # Check pairwise constraints
        ci = centers[pair_i]
        cj = centers[pair_j]
        ri = radii[pair_i]
        rj = radii[pair_j]
        dists = np.sqrt(np.sum((ci - cj)**2, axis=1))
        r_sums = ri + rj
        
        valid_pairs = r_sums > 1e-14
        if np.any(valid_pairs):
            min_scale = min(min_scale, np.min(dists[valid_pairs] / r_sums[valid_pairs]))
            
        # Apply scale with a tiny safety margin to strictly satisfy validator tolerances
        if min_scale < 1.0:
            radii *= max(0.0, min_scale - 1e-10)
            
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    return best_centers, best_radii, best_sum
