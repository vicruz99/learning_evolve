# sol_000275 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=8f5efb6b sum of radii=2.611453 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Uses SLSQP optimization starting from multiple initial configurations.
    """
    n = 26
    
    # Objective: maximize sum of radii => minimize negative sum
    def objective(vars):
        return -np.sum(vars[2::3])
    
    # Constraints:
    # 1. Boundary: r <= x <= 1-r, r <= y <= 1-r  => x-r>=0, 1-x-r>=0, etc.
    # 2. Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2 => dist_sq - sum_r_sq >= 0
    def constraints(vars):
        c = []
        for i in range(n):
            xi, yi, ri = vars[i*3], vars[i*3+1], vars[i*3+2]
            # Boundary
            c.append(xi - ri)
            c.append(1.0 - xi - ri)
            c.append(yi - ri)
            c.append(1.0 - yi - ri)
            
            # Overlap
            for j in range(i+1, n):
                xj, yj, rj = vars[j*3], vars[j*3+1], vars[j*3+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                c.append(dist_sq - sum_r*sum_r)
        return np.array(c)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * n
    
    best_result = None
    
    # Generate multiple starting configurations
    configs = []
    
    # Config 1: 5x5 Grid + 1 small circle in gap
    c1, r1 = [], []
    for r_idx in range(5):
        for c_idx in range(5):
            c1.append([0.1 + c_idx*0.2, 0.1 + r_idx*0.2])
            r1.append(0.1)
    c1.append([0.2, 0.2])
    r1.append(0.04)
    
    v1 = np.zeros(3*n)
    for i in range(26):
        v1[i*3] = c1[i][0]
        v1[i*3+1] = c1[i][1]
        v1[i*3+2] = r1[i]
    configs.append(v1)
    
    # Config 2: Hexagonal packing (dense)
    # r=0.09, spacing 2r=0.18
    r_hex = 0.09
    v2 = np.zeros(3*n)
    idx = 0
    y = r_hex
    # Pattern: 5, 4, 5, 4, 5, 4 (27 points) -> take first 26
    row_counts = [5, 4, 5, 4, 5, 4]
    for row in range(6):
        count = row_counts[row]
        # Shift for odd rows (index 1, 3, 5)
        # Even rows start at r, Odd rows start at 2r
        current_start = r_hex if row % 2 == 0 else 2*r_hex
        step = 2 * r_hex
        
        for k in range(count):
            if idx >= n: break
            x = current_start + k * step
            v2[idx*3] = x
            v2[idx*3+1] = y
            v2[idx*3+2] = r_hex
            idx += 1
        y += math.sqrt(3) * r_hex
    
    # Fill remaining if any (should be 0 for 26)
    while idx < n:
        v2[idx*3] = 0.5
        v2[idx*3+1] = 0.5
        v2[idx*3+2] = 0.01
        idx += 1
    configs.append(v2)
    
    # Config 3: Random perturbation of Config 1
    np.random.seed(42)
    v3 = configs[0] + np.random.normal(0, 0.01, configs[0].shape)
    v3[2::3] = np.clip(v3[2::3], 1e-6, 0.5)
    configs.append(v3)
    
    # Run optimization for each start
    for start in configs:
        try:
            res = opt.minimize(
                objective,
                start,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 3000, 'ftol': 1e-10}
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res
        except Exception:
            pass
            
    if best_result is None:
        best_result = opt.minimize(objective, configs[0], method='SLSQP', bounds=bounds,
                                   constraints={'type': 'ineq', 'fun': constraints})
        
    # Extract best solution
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_result.x[i*3], best_result.x[i*3+1]]
        radii[i] = best_result.x[i*3+2]
        
    # Final adjustments for numerical safety
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    # Resolve overlaps by shrinking radii if necessary (numerical precision fix)
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    # Overlap detected
                    overlap = (radii[i] + radii[j]) - dist
                    reduction = overlap / 2
                    radii[i] -= reduction
                    radii[j] -= reduction
                    radii[i] = max(radii[i], 1e-9)
                    radii[j] = max(radii[j], 1e-9)
                    changed = True
        if not changed:
            break
            
    # Re-clip centers after shrinking radii
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)

    return centers, radii, float(np.sum(radii))
