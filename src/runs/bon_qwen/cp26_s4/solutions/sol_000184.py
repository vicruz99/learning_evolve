# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 771fbeff) state=467eac2e sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization followed by SLSQP optimization.
    """
    n = 26
    
    # --- 1. Initialization: Hexagonal Packing ---
    # Start with a hexagonal grid which is dense.
    # We use a conservative radius to ensure feasibility.
    r_init = 0.085
    h = r_init * np.sqrt(3) # Vertical spacing for hex packing
    
    centers_list = []
    y = r_init
    row_idx = 0
    
    # Generate points in a hexagonal pattern until we have at least n points
    while len(centers_list) < n:
        if row_idx % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
        
        x = x_start
        while x + r_init <= 1.0 + 1e-9:
            centers_list.append([x, y])
            x += 2 * r_init
        
        y += h
        row_idx += 1
    
    # Take exactly n points
    centers = np.array(centers_list[:n])
    radii = np.full(n, r_init)
    
    # --- 2. Optimization Setup ---
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 78 variables
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i]   = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
    
    # Bounds: x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Constraints
    constraints = []
    
    # 2a. Boundary Constraints
    # x_i - r_i >= 0
    # 1.0 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1.0 - y_i - r_i >= 0
    for i in range(n):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # 1 - x >= r  => 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
        # 1 - y >= r => 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
        })

    # 2b. Non-overlap Constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # To improve performance, we only add constraints for pairs that might be close?
    # Or just add all. N=26 -> 325 pairs. It's manageable.
    # We can define a helper to create the constraint function to avoid closure issues in loops if needed,
    # but lambda with default args is safe here.
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
            
            # Using a closure-free approach via a helper or carefully constructed lambda
            # Here we pass indices i and j
            def make_overlap_constraint(i, j):
                def fun(v):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    r_sum = ri + rj
                    return dist_sq - r_sum*r_sum
                return fun

            constraints.append({
                'type': 'ineq',
                'fun': make_overlap_constraint(i, j)
            })

    # --- 3. Run Optimization ---
    # We want to maximize sum(r), so minimize -sum(r)
    def objective(v):
        r_vals = v[2::3] # Every 3rd element starting from index 2
        return -np.sum(r_vals)
    
    # Initial result
    res = None
    
    # Try to run optimization. 
    # SLSQP is a good choice for nonlinear constraints.
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    except Exception:
        # Fallback or handle error
        pass

    if res is None or not res.success:
        # If optimization failed or didn't converge well, use initial or best found
        # But usually it will return something.
        # We can try to parse the result anyway if it's not None
        if res is not None:
            x_opt = res.x
        else:
            x_opt = x0
    else:
        x_opt = res.x

    # --- 4. Extract Results ---
    centers_final = np.zeros((n, 2))
    radii_final = np.zeros(n)
    
    for i in range(n):
        centers_final[i, 0] = x_opt[3*i]
        centers_final[i, 1] = x_opt[3*i+1]
        radii_final[i] = x_opt[3*i+2]
        
        # Ensure non-negative radius (numerical safety)
        radii_final[i] = max(0.0, radii_final[i])
        # Ensure inside bounds (clip slightly if needed, though constraints should handle it)
        centers_final[i, 0] = np.clip(centers_final[i, 0], radii_final[i], 1.0 - radii_final[i])
        centers_final[i, 1] = np.clip(centers_final[i, 1], radii_final[i], 1.0 - radii_final[i])

    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii

# Helper to run locally if needed (not part of submission requirement strictly, but good for testing)
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Min radius: {np.min(r)}, Max radius: {np.max(r)}")
