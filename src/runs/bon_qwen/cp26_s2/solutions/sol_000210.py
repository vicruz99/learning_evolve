# sol_000210 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fb167b6) state=78319c72 sum of radii=2.607072 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # We arrange circles in rows with alternating counts to approximate a hexagonal packing.
    # Row counts: 5, 4, 5, 4, 5, 3 sums to 26.
    row_counts = [5, 4, 5, 4, 5, 3]
    centers = []
    
    # Initial radius for placement (small enough to avoid overlap)
    r_init = 0.05
    y_pos = r_init + 0.1 # Start slightly above bottom edge
    
    for i, count in enumerate(row_counts):
        # Horizontal spacing
        spacing = 2 * r_init
        
        if i % 2 == 0:
            # Even rows: start at left margin
            x_start = r_init
        else:
            # Odd rows: shifted by radius (hexagonal offset)
            x_start = r_init + r_init # Shift by r_init? 
            # Actually standard hex offset is r (horizontal distance between centers in adjacent rows is r)
            # If row 0 centers are at r, 3r, 5r...
            # Row 1 centers should be at 2r, 4r...
            # So x_start should be 2*r_init relative to 0?
            # Let's just center the row in [0, 1] to be safe and symmetric.
            pass
            
        # Let's place centers centered in the square for robustness
        # Total width for 'count' circles is roughly count * spacing?
        # Actually, centers are separated by 2*r.
        # Let's just use linspace for simplicity, it's a valid start.
        
        # Determine x coordinates
        # We want them roughly centered.
        # If we have 5 circles, maybe 0.1, 0.3, 0.5, 0.7, 0.9
        # If 4 circles, maybe 0.2, 0.4, 0.6, 0.8
        
        if i % 2 == 0:
            # 5 circles
            xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[:count]
        else:
            # 4 or 3 circles
            # Shifted
            if count == 4:
                xs = np.array([0.2, 0.4, 0.6, 0.8])
            elif count == 3:
                xs = np.array([0.25, 0.5, 0.75])
            else:
                # Fallback
                xs = np.linspace(0.2, 0.8, count)
        
        for x in xs:
            centers.append([x, y_pos])
        
        # Advance y by sqrt(3)/2 * diameter = sqrt(3) * r
        y_pos += np.sqrt(3) * r_init

    centers = np.array(centers)
    
    # Initial radii
    # Start with a small valid radius
    initial_radii = np.full(n, 0.04)
    
    # 2. Optimization Setup
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total variables = 26 * 3 = 78
    
    x0 = []
    for i in range(n):
        x0.append(centers[i, 0])
        x0.append(centers[i, 1])
        x0.append(initial_radii[i])
    x0 = np.array(x0)
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Objective: Minimize -sum(radii)
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)
    
    # Constraints
    # We return a vector of values that must be >= 0
    def constraints_func(vars):
        n = 26
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Boundary Constraints:
        # x >= r  => x - r >= 0
        # x <= 1-r => 1 - x - r >= 0
        # y >= r  => y - r >= 0
        # y <= 1-r => 1 - y - r >= 0
        
        c_boundaries = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        # Overlap Constraints:
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        # Vectorized computation
        # dx shape (n, n)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, None] + r[None, :]
        req_sq = r_sum**2
        
        overlap_diff = dist_sq - req_sq
        
        # We only need upper triangle (i < j) to avoid duplicates and self-checks
        # np.triu_indices(n, k=1) gives indices for strictly upper triangle
        rows, cols = np.triu_indices(n, k=1)
        c_overlap = overlap_diff[rows, cols]
        
        return np.concatenate([c_boundaries, c_overlap])

    # Define constraint for SLSQP (inequality >= 0)
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    # 3. Run Optimization
    # SLSQP is suitable for constrained non-linear problems
    # We might need multiple runs or a good start, but hex grid is usually good.
    # Increasing maxiter to ensure convergence.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success or res.fun < -2.5: # Check if we improved significantly
            final_vars = res.x
        else:
            # Fallback to initial if optimization failed (unlikely with good start)
            final_vars = x0
    except Exception:
        final_vars = x0

    # 4. Extract Results
    final_centers = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(n)])
    final_radii = final_vars[2::3]
    sum_radii = np.sum(final_radii)
    
    # Sanity check on radii (ensure non-negative)
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, sum_radii
