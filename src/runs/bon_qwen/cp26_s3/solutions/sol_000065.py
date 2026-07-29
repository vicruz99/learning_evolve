# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05a03f22) state=786e10b0 sum of radii=2.554946 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square
    to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialize with a dense hexagonal packing layout
    # Rows: 6, 5, 6, 5, 4 (Total 26)
    # This arrangement fits tightly in a hexagonal lattice.
    centers = []
    
    # Row 1: 6 circles
    row1_x = np.linspace(0.1, 0.9, 6) 
    for x in row1_x:
        centers.append([x, 0.1])
        
    # Row 2: 5 circles (staggered)
    row2_x = np.linspace(0.15, 0.85, 5)
    for x in row2_x:
        centers.append([x, 0.1 * np.sqrt(3)])
        
    # Row 3: 6 circles
    row3_y = 0.1 * np.sqrt(3) * 2
    for x in row1_x:
        centers.append([x, row3_y])
        
    # Row 4: 5 circles (staggered)
    row4_y = 0.1 * np.sqrt(3) * 3
    for x in row2_x:
        centers.append([x, row4_y])
        
    # Row 5: 4 circles (to reach 26)
    row5_x = np.linspace(0.2, 0.8, 4)
    row5_y = 0.1 * np.sqrt(3) * 4
    for x in row5_x:
        centers.append([x, row5_y])

    centers = np.array(centers)
    radii = np.full(n_circles, 0.1)
    
    # 2. Define the objective function: Negative sum of radii (to minimize)
    def objective(x_flat):
        # x_flat contains centers (n*2) followed by radii (n)
        c = x_flat[:n_circles * 2].reshape(n_circles, 2)
        r = x_flat[n_circles * 2:]
        
        return -np.sum(r)
    
    # 3. Define constraints
    constraints = []
    
    # Boundary constraints: 0 <= center - r and center + r <= 1
    for i in range(n_circles):
        x_idx = i * 2
        y_idx = i * 2 + 1
        r_idx = n_circles * 2 + i
        
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=x_idx, ri=r_idx: v[xi] - v[ri]})
        # 1 - (x + r) >= 0 => 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=x_idx, ri=r_idx: 1.0 - v[xi] - v[ri]})
        
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=y_idx, ri=r_idx: v[yi] - v[ri]})
        # 1 - (y + r) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=y_idx, ri=r_idx: 1.0 - v[yi] - v[ri]})
        
        # Non-negative radius
        constraints.append({'type': 'ineq', 'fun': lambda v, ri=r_idx: v[ri]})

    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi = i * 2, i * 2 + 1
            xj, yj = j * 2, j * 2 + 1
            ri, rj = n_circles * 2 + i, n_circles * 2 + j
            
            # dist_sq - (r_i + r_j)^2 >= 0
            def dist_constraint(v, _i=i, _j=j):
                dx = v[_i*2] - v[_j*2]
                dy = v[_i*2+1] - v[_j*2+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[n_circles*2 + _i] + v[n_circles*2 + _j]
                return dist_sq - r_sum**2
            
            constraints.append({'type': 'ineq', 'fun': dist_constraint})

    # 4. Initial vector
    x0 = np.concatenate([centers.flatten(), radii])
    
    # 5. Optimization using SLSQP
    # Bounds for coordinates [0, 1] and radius [0, 0.5]
    bounds = [(0, 1) for _ in range(n_circles * 2)] + [(0, 0.5) for _ in range(n_circles)]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 2000, 'ftol': 1e-9})
    
    # 6. Extract results
    best_centers = res.x[:n_circles * 2].reshape(n_circles, 2)
    best_radii = res.x[n_circles * 2:]
    
    # 7. Final cleanup (clip small negative values from numerical error)
    best_radii = np.maximum(best_radii, 0)
    
    # Ensure strict validity by shrinking slightly if needed (defensive)
    # Though SLSQP should handle this, we re-validate and scale down if overlap occurs
    # due to numerical precision in the optimizer.
    
    valid = validate_packing(best_centers, best_radii)
    if not valid:
        # Fallback: Scale down radii slightly until valid
        factor = 0.99
        while not validate_packing(best_centers, best_radii * factor):
            factor *= 0.99
            if factor < 0.1:
                break
        best_radii = best_radii * factor

    total_sum = np.sum(best_radii)
    return best_centers, best_radii, float(total_sum)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False

    for i in range(n):
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

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
