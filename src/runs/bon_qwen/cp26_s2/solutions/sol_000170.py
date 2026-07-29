# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 891ad575) state=a74b4bf8 sum of radii=2.606009 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square [0,1]x[0,1].
    Returns (centers, radii, sum_radii).
    """
    n = 26

    # 1. Initial Configuration: Hexagonal Lattice
    # Rows of alternating lengths (5 and 4) to utilize space.
    r_est = 0.10
    dy = math.sqrt(3) * r_est
    
    centers = []
    radii = []
    
    # Row 0 (5 circles)
    for i in range(5):
        x = r_est + i * 2 * r_est
        centers.append([x, r_est])
        radii.append(r_est)
    
    # Row 1 (4 circles, shifted)
    y1 = r_est + dy
    for i in range(4):
        x = 2 * r_est + i * 2 * r_est
        centers.append([x, y1])
        radii.append(r_est)

    # Row 2 (5 circles)
    y2 = r_est + 2 * dy
    for i in range(5):
        x = r_est + i * 2 * r_est
        centers.append([x, y2])
        radii.append(r_est)
        
    # Row 3 (5 circles, shifted x slightly or just standard)
    y3 = r_est + 3 * dy
    for i in range(5):
        x = r_est + i * 2 * r_est
        centers.append([x, y3])
        radii.append(r_est)
        
    # Row 4 (5 circles)
    y4 = r_est + 4 * dy
    for i in range(5):
        x = r_est + i * 2 * r_est
        centers.append([x, y4])
        radii.append(r_est)
        
    # Row 5 (2 circles, placed in available gaps or center)
    y5 = r_est + 5 * dy
    # Place them in the middle to balance
    centers.append([0.3, y5])
    radii.append(r_est)
    centers.append([0.7, y5])
    radii.append(r_est)

    centers = np.array(centers)
    radii = np.array(radii)

    # 2. Numerical Optimization
    # We optimize the sum of radii using coordinate descent / scipy minimize
    # Variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
    
    def objective(params):
        # Negative sum of radii for minimization
        return -np.sum(params[2::3])

    def constraint_boundary(params):
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        con = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            con.append(x - r)
            con.append(1 - (x + r))
            con.append(y - r)
            con.append(1 - (y + r))
            con.append(r) # r >= 0
        return np.array(con)

    def constraint_overlap(params):
        cons = []
        for i in range(n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                # dist^2 >= (r_i + r_j)^2
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                r_sum = ri + rj
                cons.append(dist_sq - r_sum*r_sum)
        return np.array(cons)

    # Flatten initial params
    x0 = []
    for i in range(n):
        x0.extend([centers[i][0], centers[i][1], radii[i]])
    x0 = np.array(x0)

    # Bounds for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * n

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    # Optimize
    # Use SLSQP as it handles constraints well
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})

    if res.success:
        optimal_params = res.x
        centers_opt = np.column_stack((optimal_params[0::3], optimal_params[1::3]))
        radii_opt = optimal_params[2::3]
    else:
        # Fallback to initial if optimization fails
        centers_opt = centers
        radii_opt = radii

    sum_radii = np.sum(radii_opt)
    
    # Final validation and clipping just in case
    centers_opt = np.clip(centers_opt, 1e-12, 1 - 1e-12)
    radii_opt = np.clip(radii_opt, 1e-12, None)
    
    # Re-validate constraints roughly to ensure validity for the checker
    # If any constraint is slightly violated, shrink radii
    valid = True
    for _ in range(10):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers_opt[i] - centers_opt[j])
                if dist < radii_opt[i] + radii_opt[j] - 1e-12:
                    # Violation: reduce radii
                    violation = (radii_opt[i] + radii_opt[j]) - dist
                    radii_opt[i] -= violation / 2
                    radii_opt[j] -= violation / 2
                    valid = False
        
        for i in range(n):
            if centers_opt[i][0] < radii_opt[i] + 1e-12 or centers_opt[i][0] > 1 - radii_opt[i] - 1e-12:
                radii_opt[i] = min(centers_opt[i][0], 1 - centers_opt[i][0]) - 1e-12
                valid = False
            if centers_opt[i][1] < radii_opt[i] + 1e-12 or centers_opt[i][1] > 1 - radii_opt[i] - 1e-12:
                radii_opt[i] = min(centers_opt[i][1], 1 - centers_opt[i][1]) - 1e-12
                valid = False
                
        if valid:
            break
            
    sum_radii = np.sum(radii_opt)

    return centers_opt, radii_opt, sum_radii
