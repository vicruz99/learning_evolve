# sol_000234 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=c1247390 sum of radii=2.618007 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_vars = n * 3  # x, y, r for each circle
    best_sum = -np.inf
    best_centers = None
    best_radii = None

    # Define bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(vars_flat):
        return -np.sum(vars_flat[2::3])

    # Vectorized constraint function
    def constraint_fun(vars_flat):
        c = vars_flat.reshape(-1, 3)
        # c[:, 0] = x, c[:, 1] = y, c[:, 2] = r
        constraints = []

        # Boundary constraints
        # x - r >= 0
        constraints.append(c[:, 0] - c[:, 2])
        # 1 - x - r >= 0
        constraints.append(1.0 - c[:, 0] - c[:, 2])
        # y - r >= 0
        constraints.append(c[:, 1] - c[:, 2])
        # 1 - y - r >= 0
        constraints.append(1.0 - c[:, 1] - c[:, 2])

        # Distance constraints
        # dist^2 - (r_i + r_j)^2 >= 0
        # Compute squared distance matrix
        # Centers coordinates
        coords = c[:, :2]
        radii = c[:, 2]

        # Efficiently compute pairwise distance matrix
        # (x_i - x_j)^2 + (y_i - y_j)^2
        # Using broadcasting: (n, 1, 2) - (1, n, 2)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)

        # Radius sum matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        r_sum_sq = r_sum**2

        # We only need upper triangle constraints
        # Flatten and filter
        mask = np.triu_indices(n, k=1)
        dist_constraints = dist_sq[mask] - r_sum_sq[mask]
        constraints.append(dist_constraints)

        return np.concatenate(constraints)

    # Define the single constraint dictionary for SLSQP
    # SLSQP treats 'ineq' as function value >= 0
    cons = {'type': 'ineq', 'fun': constraint_fun}

    # Helper to run optimization
    def solve(init_x0):
        try:
            res = minimize(
                objective, 
                init_x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            return res
        except Exception:
            return None

    best_res = None

    # Strategy 1: Random restarts
    for seed in range(20):
        rng = np.random.RandomState(seed)
        # Random centers
        centers = rng.rand(n, 2)
        # Small initial radii to avoid immediate violation and allow expansion
        radii = np.full(n, 0.01)
        
        x0 = np.zeros(num_vars)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
            
        res = solve(x0)
        if res is not None and res.success:
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_res = res

    # Strategy 2: Structured Hexagonal-like initialization
    # Attempt to create a loose hexagonal packing that fits easily
    # and let the optimizer tighten it.
    
    # We can fit roughly 5 rows.
    # Let's try a pattern like 5, 5, 5, 5, 6 (26 circles)
    # But 6 in a row might be tight if r is large.
    # Start with small r.
    
    r_start = 0.05
    x0_struct = np.zeros(num_vars)
    idx = 0
    
    # 5 rows
    # Distribute y uniformly
    y_coords = np.linspace(r_start, 1.0 - r_start, 5)
    
    counts = [5, 5, 5, 5, 6]
    # To make it more hexagonal, offset every other row?
    # But with small r, simple grid is fine.
    
    for row_idx, y in enumerate(y_coords):
        count = counts[row_idx]
        if count == 0: continue
        
        # Distribute x
        # Available width for centers: [r_start, 1 - r_start]
        x_min = r_start
        x_max = 1.0 - r_start
        
        if count == 1:
            xs = [0.5]
        else:
            xs = np.linspace(x_min, x_max, count)
            
        for x in xs:
            x0_struct[3*idx] = x
            x0_struct[3*idx+1] = y
            x0_struct[3*idx+2] = r_start
            idx += 1
            
    res = solve(x0_struct)
    if res is not None and res.success:
        if -res.fun > best_sum:
            best_sum = -res.fun
            best_res = res

    # Extract results
    if best_res is None:
        # Fallback to a valid grid if all else fails
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.001)
        for i in range(n):
            centers[i] = [0.5, 0.5]
        return centers, radii, 0.0

    best_x = best_res.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    # Final validation and slight adjustment for numerical safety
    # Ensure strict compliance if needed, but scipy should be precise.
    # Just return the computed values.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
