# sol_000171 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 41bc4cbf) state=caf41d6a sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses SLSQP optimization with a dense initial grid configuration.
    """
    n = 26
    
    # Define bounds for variables: x in [0, 1], y in [0, 1], r in [0, 0.5]
    # Variables layout: [x0, y0, r0, x1, y1, r1, ...]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r (max possible radius is 0.5)

    # Objective function: minimize negative sum of radii
    def objective(v):
        # Radii are at indices 2, 5, 8, ...
        return -np.sum(v[2::3])

    # Constraint functions
    # 1. Circle inside square (x bounds)
    def constr_x_lower(v):
        return v[0::3] - v[2::3]  # x - r >= 0
    
    def constr_x_upper(v):
        return 1.0 - (v[0::3] + v[2::3])  # 1 - (x + r) >= 0

    # 2. Circle inside square (y bounds)
    def constr_y_lower(v):
        return v[1::3] - v[2::3]  # y - r >= 0
    
    def constr_y_upper(v):
        return 1.0 - (v[1::3] + v[2::3])  # 1 - (y + r) >= 0

    # 3. Non-overlap between circles
    # dist^2 >= (r1 + r2)^2  =>  dist^2 - (r1 + r2)^2 >= 0
    def constr_non_overlap(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Compute squared distances for all pairs using broadcasting
        # Shape (n, n)
        dx = xs[:, None] - xs
        dy = ys[:, None] - ys
        dist2 = dx**2 + dy**2
        
        # Compute squared sum of radii
        # Shape (n, n)
        r_sum = rs[:, None] + rs[None, :]
        r_sum2 = r_sum**2
        
        # Difference
        diff = dist2 - r_sum2
        
        # Return only the upper triangle (i < j) to avoid duplicates and self-comparison
        mask = np.triu_indices(n, k=1)
        return diff[mask]

    constraints = [
        {'type': 'ineq', 'fun': constr_x_lower},
        {'type': 'ineq', 'fun': constr_x_upper},
        {'type': 'ineq', 'fun': constr_y_lower},
        {'type': 'ineq', 'fun': constr_y_upper},
        {'type': 'ineq', 'fun': constr_non_overlap}
    ]

    # Helper to flatten variables
    def to_flat(centers, radii):
        v = np.zeros(3 * n)
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        v[2::3] = radii
        return v

    # Helper to unpack variables
    def from_flat(v):
        centers = np.column_stack((v[0::3], v[1::3]))
        radii = v[2::3]
        return centers, radii

    # Best solution tracking
    best_result = None
    best_val = -np.inf

    # Try multiple starting configurations to avoid local minima
    # 1. Grid initialization (5x5 + 1)
    # 2. Random initialization
    
    starts = []
    
    # Start 1: Dense Grid
    # 5x5 grid points
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    # Add one point in a gap, e.g., center of a cell (0.2, 0.2) is distance 0.141 from corners
    # But (0.2, 0.2) might be too close to (0.1, 0.1) if r is large. 
    # Let's place it at (0.2, 0.2) but we rely on optimizer to move it.
    # Actually, better to place it at center of empty space.
    # (0.5, 0.5) is occupied.
    # Let's just pick a random point or (0.2, 0.2) with small radius.
    grid_pts.append([0.2, 0.2]) 
    
    # Take first 26
    starts.append(np.array(grid_pts[:26]))

    # Start 2: Random dense packing
    np.random.seed(123)
    rand_centers = np.random.uniform(0.1, 0.9, (n, 2))
    starts.append(rand_centers)

    # Start 3: Another random seed
    np.random.seed(456)
    rand_centers2 = np.random.uniform(0.1, 0.9, (n, 2))
    starts.append(rand_centers2)

    for centers_init in starts:
        # Initialize radii to a small valid value
        radii_init = np.ones(n) * 0.04
        
        x0 = to_flat(centers_init, radii_init)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
            
            if res.success or res.nit > 0:
                val = -res.fun # Sum of radii
                if val > best_val:
                    best_val = val
                    best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback if optimization failed
        # Return a trivial valid packing
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        # Place tiny circles
        for i in range(n):
            centers[i] = [0.5, 0.5]
            radii[i] = 0.0
        return centers, radii, 0.0

    centers, radii = from_flat(best_result.x)
    
    # Post-processing: clamp small negative radii due to numerical noise
    radii = np.maximum(radii, 0.0)
    
    # Ensure strict boundary satisfaction by shrinking slightly if needed?
    # The constraints should handle it, but just in case.
    # Check if any circle is outside
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Shrink radius if it violates boundary
        min_dist_boundary = min(x, 1-x, y, 1-y)
        if r > min_dist_boundary:
            r = min_dist_boundary
            radii[i] = r
            
    # Check overlaps and fix by shrinking
    # This is a safety net.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                req_dist = radii[i] + radii[j]
                if dist < req_dist - 1e-9:
                    # Overlap detected, reduce radii
                    overlap = req_dist - dist
                    # Reduce both radii equally or just one?
                    # Let's reduce the sum of radii to match distance
                    factor = dist / req_dist if req_dist > 0 else 0
                    radii[i] *= factor
                    radii[j] *= factor
                    changed = True

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
