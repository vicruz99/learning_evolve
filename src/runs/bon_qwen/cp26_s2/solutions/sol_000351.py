# sol_000351 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a8bfd9ed) state=88445f0f sum of radii=2.608872 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Objective: Maximize sum of radii -> Minimize -sum(radii)
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 3 * 26 = 78 variables.
    
    # We will use SLSQP.
    
    def objective(vars):
        # vars is flattened array of [x1, y1, r1, ...]
        # radii are at indices 2, 5, 8, ... (every 3rd starting from 2)
        # Actually, structure: (x, y, r) for each circle.
        # radii are at 2, 5, 8...
        radii = vars[2::3]
        return -np.sum(radii)

    def boundary_constraints(vars):
        # For each circle i:
        # r <= x <= 1-r  => x - r >= 0, 1 - x - r >= 0
        # r <= y <= 1-r  => y - r >= 0, 1 - y - r >= 0
        # r >= 0
        cons = []
        for i in range(n):
            idx = 3 * i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            cons.append(x - r)       # x >= r
            cons.append(1.0 - x - r) # x + r <= 1
            cons.append(y - r)       # y >= r
            cons.append(1.0 - y - r) # y + r <= 1
            cons.append(r)           # r >= 0
        return cons

    def overlap_constraints(vars):
        # dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                xi, yi, ri = vars[idx_i], vars[idx_i+1], vars[idx_i+2]
                xj, yj, rj = vars[idx_j], vars[idx_j+1], vars[idx_j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                
                # Constraint: dist >= sum_r  <=>  dist^2 - sum_r^2 >= 0
                # Note: This is non-convex, but SLSQP can handle it locally.
                cons.append(dist_sq - sum_r**2)
        return cons

    # Initial guess: Hexagonal packing
    # Try to fit 26 circles in a hexagonal pattern
    # Pattern: rows of 6, 5, 6, 5, 4
    centers_init = []
    radii_init = []
    
    # Estimate radius for hexagonal packing
    # 5 rows. Height ~ 2r + 4*r*sqrt(3). Width ~ 6*2r (for row of 6).
    # Let's try to fit in 1x1.
    # If we fit 6 circles in width, r ~ 1/12 = 0.0833.
    # If we fit 5 circles in width, r ~ 1/10 = 0.1.
    # Hexagonal packing is denser, maybe r ~ 0.09?
    # Let's start with r = 0.09.
    
    r_est = 0.09
    # Coordinates
    row_counts = [6, 5, 6, 5, 4] # Total 26
    y = r_est
    for count in row_counts:
        # Center the row
        # Total width occupied by 'count' circles is (count-1)*2r + 2r = 2r*count ?
        # No, width spanned is 2r * count.
        # To center in [0, 1], start x = (1 - 2r*count)/2
        start_x = (1.0 - 2.0 * r_est * count) / 2.0
        for k in range(count):
            x = start_x + k * 2.0 * r_est
            centers_init.append([x, y])
            radii_init.append(r_est)
        
        # Offset next row in x by r_est for hexagonal packing?
        # Actually, in standard hex, rows are shifted by r.
        # But our row counts alternate, so shifting might align them nicely?
        # Let's just stack vertically with vertical spacing r*sqrt(3)
        y += r_est * math.sqrt(3)

    vars_init = []
    for i in range(n):
        vars_init.extend(centers_init[i])
        vars_init.append(radii_init[i])
    vars_init = np.array(vars_init)

    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bnds = []
    for i in range(n):
        bnds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Constraints dict
    # SLSQP expects constraints as dict or list of dicts
    # Nonlinear constraints: type 'ineq' means expression >= 0
    
    # We can combine boundary and overlap constraints
    def all_constraints(vars):
        # Boundary
        b_cons = boundary_constraints(vars)
        # Overlap
        o_cons = overlap_constraints(vars)
        return b_cons + o_cons

    # Since constraints are many, maybe split?
    # SLSQP can take a list of constraints.
    
    # To make it faster, we might optimize equal radii first?
    # But let's try full optimization. 78 vars is okay for SLSQP.
    
    best_sum = -np.inf
    best_vars = None
    
    # Run multiple restarts
    num_restarts = 5
    
    for seed in range(num_restarts):
        np.random.seed(seed)
        
        # Perturb initial guess slightly
        current_vars = vars_init.copy()
        if seed > 0:
            # Add noise
            noise = np.random.normal(0, 0.01, size=current_vars.shape)
            current_vars += noise
            # Clip to bounds
            for i in range(n):
                current_vars[3*i] = np.clip(current_vars[3*i], 0, 1)
                current_vars[3*i+1] = np.clip(current_vars[3*i+1], 0, 1)
                current_vars[3*i+2] = np.clip(current_vars[3*i+2], 0, 0.5)

        try:
            res = opt.minimize(objective, 
                               current_vars, 
                               method='SLSQP', 
                               bounds=bnds,
                               constraints={'type': 'ineq', 'fun': all_constraints},
                               options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or (not np.isnan(res.fun)):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_vars = res.x
        except Exception as e:
            print(f"Optimization failed at seed {seed}: {e}")
            continue

    if best_vars is None:
        # Fallback to initial guess sum
        return np.array(centers_init), np.array(radii_init), np.sum(radii_init)

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_vars[3*i], best_vars[3*i+1]]
        radii[i] = best_vars[3*i+2]

    # Final validation check (just in case)
    # The optimizer might return a point slightly violating constraints due to tolerance
    # But we should return the best found.
    
    return centers, radii, float(best_sum)
