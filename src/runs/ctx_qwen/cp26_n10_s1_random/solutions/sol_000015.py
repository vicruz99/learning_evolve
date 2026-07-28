# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f64c520b) state=9bb7cb0f sum of radii=1.937479 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_vars = n * 3
    
    # Initial guess parameters
    # r=0.09 allows fitting > 26 circles in a hex grid, ensuring a valid start.
    initial_r = 0.09
    
    # Generate initial hexagonal grid points
    points = []
    row = 0
    y = initial_r
    row_height = initial_r * np.sqrt(3)
    
    # Fill grid within [0, 1] x [0, 1]
    while y + initial_r <= 1.0:
        shift = 0.0
        if row % 2 == 1:
            shift = initial_r
        
        x = initial_r + shift
        while x + initial_r <= 1.0:
            points.append([x, y])
            x += 2 * initial_r
        
        y += row_height
        row += 1
    
    # Fallback if not enough points generated
    if len(points) < n:
        for i in range(n):
            col = i % 6
            row_idx = i // 6
            points.append([0.1 + col * 0.15, 0.1 + row_idx * 0.15])
            
    points = points[:n]
    centers = np.array(points)
    radii = np.full(n, initial_r)
    
    # Flatten to 1D vector for optimization: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(num_vars)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i + 1] = centers[i, 1]
        x0[3*i + 2] = radii[i]
    
    # Bounds: x, y in [0, 1], r in [small_positive, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n

    # Penalty weight for constraint violation
    penalty_weight = 5000.0

    def objective(vars):
        c_x = vars[0::3]
        c_y = vars[1::3]
        r = vars[2::3]
        
        # Objective: maximize sum of radii => minimize negative sum
        obj = -np.sum(r)
        penalty = 0.0
        
        # Overlap penalty: sum of squared violations
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (c_x[i] - c_x[j])**2 + (c_y[i] - c_y[j])**2
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    violation = min_dist - dist
                    penalty += violation**2
        
        # Boundary penalty: sum of squared violations
        for i in range(n):
            # Left boundary: x - r >= 0
            val = c_x[i] - r[i]
            if val < 0: penalty += val**2
            
            # Right boundary: x + r <= 1
            val = 1.0 - (c_x[i] + r[i])
            if val < 0: penalty += val**2
            
            # Bottom boundary: y - r >= 0
            val = c_y[i] - r[i]
            if val < 0: penalty += val**2
            
            # Top boundary: y + r <= 1
            val = 1.0 - (c_y[i] + r[i])
            if val < 0: penalty += val**2
                
        return obj + penalty_weight * penalty

    # Run optimization
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-15})
    
    # Extract results
    final_vars = res.x
    final_centers = np.column_stack((final_vars[0::3], final_vars[1::3]))
    final_radii = final_vars[2::3]
    
    # Post-processing: Scale radii down slightly to ensure strict validity
    # Calculate maximum scaling factor k such that k*radii satisfies all constraints
    k = 1.0
    
    # Check boundary constraints
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        if r < 1e-9: continue
        
        # x >= k*r => k <= x/r
        ratio = x / r
        if ratio < k: k = ratio
        
        # 1 - x >= k*r => k <= (1-x)/r
        ratio = (1.0 - x) / r
        if ratio < k: k = ratio
        
        # y >= k*r => k <= y/r
        ratio = y / r
        if ratio < k: k = ratio
        
        # 1 - y >= k*r => k <= (1-y)/r
        ratio = (1.0 - y) / r
        if ratio < k: k = ratio
        
    # Check pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
            r_sum = final_radii[i] + final_radii[j]
            if r_sum > 1e-9:
                ratio = dist / r_sum
                if ratio < k:
                    k = ratio
    
    # Apply scaling with safety margin for numerical precision
    k = k * 0.99999
    final_radii *= k
    final_sum = np.sum(final_radii)
    
    return final_centers, final_radii, final_sum
