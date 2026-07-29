# sol_000206 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=93cef6fc sum of radii=2.589504 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Solves the circle packing problem for 26 circles in a unit square 
    to maximize the sum of radii.
    """
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Vectorized constraint function for SLSQP
    def constraint_func(v):
        # Extract variables
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        # All must be >= 0
        cons_boundary = np.concatenate([
            xs - rs,
            1.0 - xs - rs,
            ys - rs,
            1.0 - ys - rs
        ])
        
        # Overlap constraints: dist^2 >= (ri + rj)^2
        # Vectorized calculation for all pairs (i, j) where i < j
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = rs[:, None] + rs[None, :]
        r_sum_sq = r_sum**2
        
        # Compute constraint values: dist_sq - r_sum_sq
        cons_matrix = dist_sq - r_sum_sq
        
        # Extract strictly upper triangular part (unique pairs)
        # mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        cons_overlap = cons_matrix[mask]
        
        return np.concatenate([cons_boundary, cons_overlap])

    # Objective function: Minimize negative sum of radii
    def objective(v):
        return -np.sum(v[2::3])

    # Bounds for x, y in [0, 1] and r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Constraint definition
    cons = {'type': 'ineq', 'fun': constraint_func}

    # Optimization Options
    opts = {'maxiter': 500, 'ftol': 1e-12, 'disp': False}

    def run_optimization(v0):
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options=opts)
        return res

    def get_hex_init():
        v = np.zeros(3 * n)
        # Hexagonal grid parameters
        spacing = 0.16 # Initial spacing, roughly 2 * r
        idx = 0
        
        # Iterate rows
        y = spacing
        row = 0
        while y <= 1.0 and idx < n:
            shift = (row % 2) * (spacing / 2)
            x = spacing + shift
            while x <= 1.0 and idx < n:
                v[3*idx] = x
                v[3*idx+1] = y
                v[3*idx+2] = 0.08 # Initial small radius
                idx += 1
                x += spacing
            y += spacing * np.sqrt(3) / 2
            row += 1
            
        # Fill any remaining slots if loop ended early (shouldn't with these params)
        while idx < n:
            v[3*idx] = 0.5
            v[3*idx+1] = 0.5
            v[3*idx+2] = 0.08
            idx += 1
        return v

    # 1. Hexagonal Initialization
    best_sum = -1
    v0_hex = get_hex_init()
    res = run_optimization(v0_hex)
    if res.success:
        best_sum = -res.fun
        best_centers = np.array([res.x[3*i:3*i+2] for i in range(n)])
        best_radii = res.x[2::3]

    # 2. Random Restarts to escape local minima
    np.random.seed(42)
    for _ in range(5):
        # Random centers and small radii
        v0_rand = np.zeros(3 * n)
        v0_rand[0::3] = np.random.rand(n) * 0.8 + 0.1 # Keep away from edges initially
        v0_rand[1::3] = np.random.rand(n) * 0.8 + 0.1
        v0_rand[2::3] = 0.02
        
        # Shuffle centers to avoid symmetry
        rng = np.random.default_rng()
        rng.shuffle(v0_rand[0::3])
        rng.shuffle(v0_rand[1::3])
        
        res = run_optimization(v0_rand)
        if res.success:
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = np.array([res.x[3*i:3*i+2] for i in range(n)])
                best_radii = res.x[2::3]

    # Post-processing: Ensure strict validity
    if best_centers is not None:
        # Clip centers to [0, 1]
        best_centers = np.clip(best_centers, 0.0, 1.0)
        
        # Clip radii to be non-negative
        best_radii = np.maximum(best_radii, 0.0)
        
        # Enforce boundary constraints strictly: r <= min(x, 1-x, y, 1-y)
        for i in range(n):
            x, y = best_centers[i]
            margin = min(x, 1.0 - x, y, 1.0 - y)
            if best_radii[i] > margin:
                best_radii[i] = margin
        
        # Final overlap resolution (reduce radii if slightly overlapping)
        # This handles floating point inaccuracies from the solver
        # We iterate to reduce the radii of overlapping pairs
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                    req_sum = best_radii[i] + best_radii[j]
                    if req_sum > dist + 1e-12:
                        # Overlap detected, shrink radii proportionally
                        diff = req_sum - dist
                        # Distribute reduction equally
                        reduction = diff / 2.0
                        best_radii[i] -= reduction
                        best_radii[j] -= reduction
                        best_radii = np.maximum(best_radii, 0.0)
                        changed = True

        best_sum = float(np.sum(best_radii))
    else:
        # Fallback: small circles in a grid
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.01
        for i in range(n):
            r = i // 10
            c = i % 10
            centers[i] = [0.05 + c * 0.09, 0.05 + r * 0.09]
        best_centers = centers
        best_radii = radii
        best_sum = 0.26

    return best_centers, best_radii, best_sum
