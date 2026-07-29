# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e52471dd) state=b91b0bbc sum of radii=2.618046 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    
    # 1. Hexagonal Lattice Initialization
    # We attempt to create a hexagonal grid that covers the unit square.
    # Row spacing (vertical) = sqrt(3) * r
    # Col spacing (horizontal) = 2 * r
    # Shifted rows by r.
    
    # Initial guess for radius. For 26 circles, a radius slightly below 0.1 is a good start.
    r_init = 0.09 
    
    # Generate points
    points = []
    
    # Estimate number of rows
    # Height needed for k rows: 2*r + (k-1)*sqrt(3)*r <= 1
    # k approx 1 + (1-2r)/(sqrt(3)r)
    num_rows = int(1 + (1 - 2*r_init) / (np.sqrt(3) * r_init)) + 2
    
    row_height = np.sqrt(3) * r_init
    
    for row in range(num_rows):
        y = r_init + row * row_height
        if y + r_init > 1.0 + 1e-5:
            break
            
        # Determine x-step and start offset based on row index (even/odd)
        if row % 2 == 0:
            x_start = r_init
            x_step = 2 * r_init
        else:
            x_start = 2 * r_init # Shifted by r relative to even rows (since even starts at r)
            # Actually, if even starts at r, odd centers are at r + r = 2r? 
            # Distance between (r, y) and (2r, y+dy) is sqrt(r^2 + 3r^2) = 2r. Correct.
            x_step = 2 * r_init
            
        x = x_start
        while x + r_init <= 1.0 + 1e-5:
            points.append((x, y))
            x += x_step
            
    # We need exactly 26 circles.
    # If we have more, we take the first 26 (top-left usually dense).
    # If fewer, the optimizer will try to spread them out, but we should have enough.
    if len(points) > n_circles:
        points = points[:n_circles]
    elif len(points) < n_circles:
        # Fallback: add points in gaps or just random if grid failed (unlikely)
        # Pad with points near center if needed
        while len(points) < n_circles:
            points.append((0.5, 0.5))

    # Convert to numpy array
    centers_init = np.array(points[:n_circles])
    
    # Optimization Setup
    def objective(vars_):
        # vars_ structure: [x1, y1, r1, x2, y2, r2, ...]
        # But easier to handle constraints if we split
        # Let's use flat array: centers (2*n) then radii (n)
        # Actually, mixing them is fine.
        # vars_[0] = x1, vars_[1] = y1, vars_[2] = r1 ...
        # No, let's do [x1..xn, y1..yn, r1..rn] for easier slicing?
        # Or [x1, y1, r1, ...]
        # Let's use [x1, y1, r1, x2, y2, r2, ...] -> size 3*n
        radii = vars_[2::3]
        return -np.sum(radii) # Minimize negative sum = Maximize sum

    def boundary_constraints(vars_):
        c = []
        for i in range(n_circles):
            idx = 3 * i
            x = vars_[idx]
            y = vars_[idx+1]
            r = vars_[idx+2]
            
            # x >= r
            c.append(x - r)
            # x <= 1 - r  => x + r <= 1
            c.append(1.0 - (x + r))
            # y >= r
            c.append(y - r)
            # y <= 1 - r
            c.append(1.0 - (y + r))
            # r >= 0 (handled by bounds or just constraint)
            c.append(r)
        return np.array(c)

    def overlap_constraints(vars_):
        c = []
        for i in range(n_circles):
            idx_i = 3 * i
            xi, yi, ri = vars_[idx_i], vars_[idx_i+1], vars_[idx_i+2]
            for j in range(i + 1, n_circles):
                idx_j = 3 * j
                xj, yj, rj = vars_[idx_j], vars_[idx_j+1], vars_[idx_j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist = ri + rj
                # Constraint: dist >= ri + rj  => dist^2 >= (ri+rj)^2
                # Non-convex, but SLSQP handles it.
                # Using dist^2 - (ri+rj)^2 >= 0
                c.append(dist_sq - min_dist**2)
        return np.array(c)

    # Initial guess vector
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init # Start with equal radii

    # Bounds for radii: [0, 0.5] (max radius in unit square)
    # Bounds for centers: [0, 1]
    bounds = []
    for i in range(n_circles):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Constraints dictionary
    cons = []
    
    # Boundary constraints (g(x) >= 0)
    cons.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Overlap constraints (g(x) >= 0)
    cons.append({'type': 'ineq', 'fun': overlap_constraints})

    # Multi-start optimization to find a better global optimum
    best_sol = None
    best_obj = -np.inf
    
    # Run optimization a few times with slightly perturbed starts
    n_trials = 5
    
    for trial in range(n_trials):
        # Perturb initial positions slightly
        perturbation = np.random.normal(0, 0.005, size=x0.shape)
        # Ensure bounds are respected after perturbation
        x_trial = x0 + perturbation
        # Clip centers to [0.1, 0.9] to be safe, radii to [0.05, 0.15]
        for i in range(n_circles):
            x_trial[3*i] = np.clip(x_trial[3*i], 0.01, 0.99)
            x_trial[3*i+1] = np.clip(x_trial[3*i+1], 0.01, 0.99)
            x_trial[3*i+2] = np.clip(x_trial[3*i+2], 0.05, 0.15)

        try:
            res = minimize(
                objective,
                x_trial,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success or (res.fun < -np.inf + 1e-3): # Check if valid
                current_sum = -res.fun
                if current_sum > best_obj:
                    best_obj = current_sum
                    best_sol = res.x
        except Exception:
            continue

    if best_sol is None:
        # Fallback to original guess if optimization fails
        best_sol = x0

    # Extract results
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        centers[i, 0] = best_sol[3*i]
        centers[i, 1] = best_sol[3*i+1]
        radii[i] = best_sol[3*i+2]

    # Post-processing: Ensure strict validity (handle floating point errors)
    # Scale down slightly if needed
    min_margin = 1e-10
    
    # Check boundaries
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        # Ensure inside [0,1]
        centers[i, 0] = np.clip(x, r + min_margin, 1.0 - r - min_margin)
        centers[i, 1] = np.clip(y, r + min_margin, 1.0 - r - min_margin)
        # If r is too large for the clipped center, reduce r
        max_r_x = 1.0 - centers[i, 0]
        max_r_y = 1.0 - centers[i, 1]
        min_r_bound = min(centers[i, 0], centers[i, 1])
        max_allowed_r = min(max_r_x, max_r_y, min_r_bound) - min_margin
        if radii[i] > max_allowed_r:
            radii[i] = max_allowed_r

    # Check overlaps and shrink if necessary
    # Iterative shrinking
    for _ in range(10):
        max_violation = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req_dist = radii[i] + radii[j]
                if dist < req_dist - 1e-12:
                    # Violation
                    violation = req_dist - dist
                    # Shrink radii equally to fix violation
                    shrink = violation / 2.0 + 1e-6
                    radii[i] -= shrink
                    radii[j] -= shrink
        
        # Ensure radii non-negative
        radii = np.maximum(radii, 0)
        
        # If no violation, break
        overlap_ok = True
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    overlap_ok = False
                    break
            if not overlap_ok:
                break
        if overlap_ok:
            break

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
