# sol_000310 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39d28b7b) state=7be1c896 sum of radii=2.594686 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Helper to create hexagonal lattice initialization
    def get_hexagonal_init(n, scale=0.09, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Estimate grid size
        # Hexagonal packing density allows fitting more circles.
        # Try to fit in roughly 6 rows or 5 rows.
        # For 26 circles, a 6-row arrangement (5,4,5,4,5,3) or similar works well.
        # Or just a rectangular hexagonal grid.
        
        cols = 6
        rows = 5
        
        centers = []
        for r in range(rows):
            for c in range(cols):
                if len(centers) >= n:
                    break
                # Hexagonal spacing
                x = c * 2 * scale + (r % 2) * scale
                y = r * scale * np.sqrt(3)
                centers.append([x, y])
            if len(centers) >= n:
                break
        
        # If we didn't get enough points (unlikely with these params), pad with random
        while len(centers) < n:
            centers.append([np.random.rand(), np.random.rand()])
            
        centers = np.array(centers[:n])
        
        # Center the configuration in the unit square
        # Current bounds
        x_min, y_min = centers.min(axis=0)
        x_max, y_max = centers.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        
        # Desired bounds (leave some margin for radii)
        # If radii are approx scale, margin should be scale.
        margin = scale * 1.5
        target_w, target_h = 1 - 2*margin, 1 - 2*margin
        
        # Scale and shift
        if w > 0 and h > 0:
            sx = target_w / w
            sy = target_h / h
            s = min(sx, sy)
            
            centers[:, 0] = centers[:, 0] * s
            centers[:, 1] = centers[:, 1] * s
            
            # Shift to center
            cx, cy = centers.mean(axis=0)
            centers[:, 0] += 0.5 - cx
            centers[:, 1] += 0.5 - cy
            
            # Clip to valid range [margin, 1-margin] just in case
            centers[:, 0] = np.clip(centers[:, 0], margin, 1 - margin)
            centers[:, 1] = np.clip(centers[:, 1], margin, 1 - margin)

        radii = np.full(n, scale)
        return centers, radii

    def objective(variables):
        # variables: [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum(r), so minimize -sum(r)
        radii = variables[2::3]
        return -np.sum(radii)

    def boundary_constraints(variables):
        # x >= r, x <= 1-r => r - x <= 0, x + r - 1 <= 0
        # Same for y
        cons = []
        for i in range(n):
            idx = 3 * i
            x, y, r = variables[idx], variables[idx+1], variables[idx+2]
            cons.append(r - x)       # r - x <= 0 => x >= r
            cons.append(x + r - 1)   # x + r - 1 <= 0 => x <= 1-r
            cons.append(r - y)       # r - y <= 0 => y >= r
            cons.append(y + r - 1)   # y + r - 1 <= 0 => y <= 1-r
        return np.array(cons)

    def overlap_constraints(variables):
        # dist >= r1 + r2 => (r1 + r2)^2 - dist^2 <= 0
        # dist^2 = (x1-x2)^2 + (y1-y2)^2
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                xi, yi, ri = variables[idx_i], variables[idx_i+1], variables[idx_i+2]
                xj, yj, rj = variables[idx_j], variables[idx_j+1], variables[idx_j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                radius_sum = ri + rj
                cons.append(radius_sum**2 - dist_sq)
        return np.array(cons)

    # Prepare optimization
    # We will run multiple starts
    best_sum = -1
    best_centers = None
    best_radii = None
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Define constraints for SLSQP
    # SLSQP expects inequality constraints g(x) <= 0
    cons = []
    
    # Boundary constraints: r - x <= 0, etc.
    # We can't easily vectorize the list of 4*n constraints for SLSQP callback without a function
    # But SLSQP allows 'type': 'ineq' with a function that returns array.
    
    def con_bounds(val):
        # Returns array of values that must be >= 0? 
        # SLSQP 'ineq' means function(x) >= 0.
        # My formulation above was <= 0. Let's flip signs.
        # x - r >= 0, 1 - x - r >= 0, ...
        res = []
        for i in range(n):
            idx = 3 * i
            x, y, r = val[idx], val[idx+1], val[idx+2]
            res.append(x - r)
            res.append(1 - x - r)
            res.append(y - r)
            res.append(1 - y - r)
        return np.array(res)

    def con_overlap(val):
        # dist^2 - (r1+r2)^2 >= 0
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                xi, yi, ri = val[idx_i], val[idx_i+1], val[idx_i+2]
                xj, yj, rj = val[idx_j], val[idx_j+1], val[idx_j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                radius_sum = ri + rj
                res.append(dist_sq - radius_sum**2)
        return np.array(res)

    constraints = [
        {'type': 'ineq', 'fun': con_bounds},
        {'type': 'ineq', 'fun': con_overlap}
    ]

    # Try a few random seeds to find a good local optimum
    seeds = [42, 123, 456, 789, 1000]
    
    for seed in seeds:
        # Initialize
        # Use a slightly larger scale to encourage larger radii, but ensure validity is possible
        # Hexagonal init with scale 0.09 is safe.
        # Let's try to initialize with a radius close to what we expect (0.1)
        # But we must ensure initial configuration is valid or close to valid.
        # A hexagonal lattice with r=0.09 is valid.
        
        init_centers, init_radii = get_hexagonal_init(n, scale=0.09, random_seed=seed)
        
        # Flatten to 1D vector
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = init_centers[i, 0]
            x0[3*i+1] = init_centers[i, 1]
            x0[3*i+2] = init_radii[i]
            
        # Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success or res.fun < 0: # fun is -sum_radii
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    # Extract solution
                    best_centers = np.zeros((n, 2))
                    best_radii = np.zeros(n)
                    for i in range(n):
                        best_centers[i, 0] = res.x[3*i]
                        best_centers[i, 1] = res.x[3*i+1]
                        best_radii[i] = res.x[3*i+2]
        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")
            continue

    # Fallback if optimization fails completely (unlikely)
    if best_centers is None:
        best_centers, best_radii = get_hexagonal_init(n, scale=0.09, random_seed=42)
        best_sum = np.sum(best_radii)

    # Ensure radii are non-negative (clipping might be needed due to numerical errors)
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, best_sum
