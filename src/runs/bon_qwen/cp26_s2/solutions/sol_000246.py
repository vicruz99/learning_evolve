# sol_000246 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8f88c46f) state=041d5f63 sum of radii=2.495107 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26

    # --- Strategy: Optimize for Equal Radii ---
    # We optimize the radius r and centers (x, y) to maximize r.
    # Total variables: 26 centers * 2 coords + 1 radius = 53 variables.
    # However, SLSQP handles inequality constraints well. 
    # To make it faster, we can fix r and optimize centers, or optimize all.
    # Here we optimize centers and r together.

    # 1. Initialization: Hexagonal Grid
    # Pattern: 5, 4, 5, 4, 5, 3 circles in rows
    r_init = 0.09
    centers = []
    rows = [5, 4, 5, 4, 5, 3]
    y_curr = r_init
    
    # Pre-calculate horizontal offsets for centered placement
    # For n circles of radius r, centers are at: 
    # [r, r+2r, ..., r+(n-1)2r] -> width is 2r*n.
    # To center in [0,1], shift by (1 - 2r*n)/2.
    
    for count in rows:
        width_row = 2 * r_init * count
        shift = (1.0 - width_row) / 2.0
        for i in range(count):
            x = shift + r_init + i * (2 * r_init)
            centers.append([x, y_curr])
        
        # Move to next row (hexagonal vertical spacing)
        if count > 0:
            y_curr += np.sqrt(3) * r_init # Vertical distance between rows

    centers = np.array(centers)
    # Ensure we have 26 centers
    assert len(centers) == n, f"Initialization failed: got {len(centers)} centers"

    # 2. Define Objective and Constraints
    # We minimize negative radius (to maximize it).
    # Variables vector: [x1, y1, x2, y2, ..., x26, y26, r]
    # Length = 2*n + 1

    def objective(vars_arr):
        # We want to maximize r, so minimize -r
        # r is the last element
        return -vars_arr[-1]

    def boundary_constraints(vars_arr):
        r = vars_arr[-1]
        c = vars_arr[:-1].reshape(-1, 2)
        cons = []
        for i in range(n):
            # x >= r
            cons.append(c[i, 0] - r)
            # x <= 1 - r
            cons.append(1.0 - c[i, 0] - r)
            # y >= r
            cons.append(c[i, 1] - r)
            # y <= 1 - r
            cons.append(1.0 - c[i, 1] - r)
        return np.array(cons)

    def non_overlap_constraints(vars_arr):
        r = vars_arr[-1]
        c = vars_arr[:-1].reshape(-1, 2)
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                # dist^2 >= (2r)^2
                # (xi - xj)^2 + (yi - yj)^2 >= 4r^2
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_sq = dx**2 + dy**2
                min_dist_sq = (2 * r)**2
                cons.append(dist_sq - min_dist_sq)
        return np.array(cons)

    # Initial guess vector
    x0_centers = centers.flatten()
    x0_r = r_init
    x0 = np.concatenate([x0_centers, [x0_r]])

    # Bounds: r must be positive, coordinates in [0,1]
    # Simple bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(2 * n):
        bounds.append((0, 1))
    bounds.append((0.01, 0.5)) # r must be > 0

    # Setup constraints for SLSQP
    # SLSQP expects constraints as dictionaries or non-linear functions
    # We will combine them into one function returning an array for efficiency, 
    # but SLSQP needs dict format for 'ineq' constraints usually, or we can pass a callable.
    # Actually, standard scipy.optimize.minimize with method='SLSQP' accepts 
    # constraints list where each item is a dict with 'type' and 'fun'.
    
    # To reduce overhead, we can define one big constraint function for boundaries
    # and one for overlaps.
    
    constraint_boundaries = {
        'type': 'ineq',
        'fun': boundary_constraints
    }
    
    constraint_overlaps = {
        'type': 'ineq',
        'fun': non_overlap_constraints
    }

    constraints = [constraint_boundaries, constraint_overlaps]

    # 3. Run Optimization
    # Using SLSQP. Max iterations increased.
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )

    # 4. Extract Results
    if res.success or res.fun > -100: # Check if optimization didn't fail catastrophically
        opt_centers = res.x[:-1].reshape(-1, 2)
        opt_r = res.x[-1]
        
        # Ensure radii are non-negative (though bounds should handle it)
        opt_r = max(0.0, opt_r)
        
        radii = np.full(n, opt_r)
        
        # Calculate sum
        sum_radii = np.sum(radii)
        
        return opt_centers, radii, sum_radii
    else:
        # Fallback to initial configuration if optimization fails
        radii = np.full(n, r_init)
        sum_radii = np.sum(radii)
        return centers, radii, sum_radii
