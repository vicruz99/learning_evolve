# sol_000265 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=ae0f9432 sum of radii=2.604260 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    n = 26
    
    # --- Step 1: Initialization ---
    # We start with a heuristic hexagonal grid pattern.
    # Pattern: 5, 4, 5, 4, 5, 3 circles in rows (Total 26).
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    row_counts = [5, 4, 5, 4, 5, 3]
    idx = 0
    
    # Y-coordinates for rows, distributed in [0.15, 0.85] to allow margin
    y_vals = np.linspace(0.15, 0.85, 6)
    
    for i, count in enumerate(row_counts):
        y = y_vals[i]
        
        # X coordinates: alternate shifted rows for hexagonal packing
        if i % 2 == 1:
            # Shifted row
            if count == 4:
                # Place 4 circles in gaps of 5-circle row
                xs = np.array([0.2, 0.4, 0.6, 0.8])
            else:
                xs = np.linspace(0.1, 0.9, count)
        else:
            # Standard row
            if count == 5:
                xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            else:
                xs = np.linspace(0.1, 0.9, count)
        
        for x in xs:
            centers[idx] = [x, y]
            idx += 1
            
    # Initial radii: 0.05 (safe, valid configuration)
    radii[:] = 0.05
    
    # --- Step 2: Optimization ---
    # We use a penalty method to maximize sum of radii.
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    x0 = np.zeros(3 * n)
    for k in range(n):
        x0[3*k] = centers[k, 0]
        x0[3*k+1] = centers[k, 1]
        x0[3*k+2] = 0.05
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    def objective(params):
        # Objective: Minimize -sum(radii) + Penalty
        sum_r = -sum(params[3*k+2] for k in range(n))
        
        penalty = 0.0
        lam = 500.0  # Penalty weight
        
        # Boundary penalties
        for k in range(n):
            x, y, r = params[3*k], params[3*k+1], params[3*k+2]
            # Check x bounds: r <= x <= 1-r
            if x < r: penalty += (x - r)**2
            if x > 1 - r: penalty += (x - (1 - r))**2
            # Check y bounds: r <= y <= 1-r
            if y < r: penalty += (y - r)**2
            if y > 1 - r: penalty += (y - (1 - r))**2
            
        # Pairwise distance penalties
        # Constraint: dist^2 >= (r_i + r_j)^2
        for i in range(n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (ri + rj)**2
                
                if min_dist_sq > dist_sq:
                    penalty += (min_dist_sq - dist_sq)**2
                    
        return sum_r + lam * penalty

    # Optimize using L-BFGS-B
    res = opt.minimize(
        fun=objective,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    # --- Step 3: Post-processing ---
    # Extract solution and strictly enforce constraints
    best_params = res.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for k in range(n):
        final_centers[k, 0] = best_params[3*k]
        final_centers[k, 1] = best_params[3*k+1]
        final_radii[k] = best_params[3*k+2]
        
    # Iterative shrinking to resolve any residual violations
    for _ in range(200):
        changed = False
        
        # Boundary constraints
        for k in range(n):
            x, y = final_centers[k]
            r = final_radii[k]
            max_r = min(x, 1 - x, y, 1 - y)
            if r > max_r + 1e-12:
                final_radii[k] = max_r
                changed = True
        
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = final_centers[i, 0] - final_centers[j, 0]
                dy = final_centers[i, 1] - final_centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = final_radii[i] + final_radii[j]
                
                if dist < r_sum - 1e-12:
                    # Overlap detected, reduce radii equally
                    excess = r_sum - dist
                    final_radii[i] -= excess / 2
                    final_radii[j] -= excess / 2
                    changed = True
        
        if not changed:
            break
            
    final_radii = np.maximum(final_radii, 0.0)
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
