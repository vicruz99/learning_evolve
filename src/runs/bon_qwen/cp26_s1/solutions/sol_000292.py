# sol_000292 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 86cff419) state=ab95912d sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    np.random.seed(42)  # For reproducibility

    # 1. Initialization: Hexagonal Lattice
    # Hexagonal packing is denser than square grid.
    # We estimate a diameter that might fit. 
    # For n=26, r ~ 0.1 is a good estimate. Let's start slightly smaller to be safe.
    r_init = 0.08
    diameter = 2 * r_init
    
    centers = []
    row = 0
    col = 0
    while len(centers) < n:
        # Hexagonal offset
        x = col * diameter + (row % 2) * (diameter / 2) + r_init
        y = row * (diameter * np.sqrt(3) / 2) + r_init
        
        # Check if inside square (with some margin)
        if x < 1 - r_init and y < 1 - r_init:
            centers.append([x, y])
        
        col += 1
        if col * diameter > 1:
            col = 0
            row += 1

    # If we didn't fill 26 (unlikely with this logic but safe check), fill remaining randomly
    while len(centers) < n:
        centers.append([np.random.uniform(r_init, 1-r_init), np.random.uniform(r_init, 1-r_init)])
    
    centers = np.array(centers[:n])
    radii = np.full(n, r_init)
    
    best_sum_radii = -1
    best_centers = None
    best_radii = None

    # 2. Optimization Loop with multiple restarts
    for restart in range(10):
        # Perturb initial radii slightly
        current_radii = radii * (1 + np.random.uniform(-0.1, 0.1, n))
        # Ensure radii are positive and not too large initially
        current_radii = np.clip(current_radii, 0.01, 0.5)
        
        # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.hstack([centers.flatten(), current_radii])

        def objective(vars_flat):
            r_vec = vars_flat[2::3]
            return -np.sum(r_vec) # Maximize sum of radii

        # Constraints
        constraints = []

        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x_i - r_i >= 0  => vars[3*i] - vars[3*i+2] >= 0
        # 1 - x_i - r_i >= 0 => 1 - vars[3*i] - vars[3*i+2] >= 0
        # Same for y
        for i in range(n):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx_x] - v[idx_r]
            })
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
            })
            # y >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx_y] - v[idx_r]
            })
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
            })

        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
                idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
                
                def dist_constraint(v, i=i, j=j):
                    xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                    xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                    dx = xi - xj
                    dy = yi - yj
                    return dx*dx + dy*dy - (ri + rj)**2
                
                constraints.append({
                    'type': 'ineq',
                    'fun': dist_constraint
                })

        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_centers = res.x[0::3].reshape(n, 2)
                    best_radii = res.x[2::3]
        except Exception as e:
            continue

    # If optimization failed to find better than init (unlikely), return init or best found
    if best_centers is None:
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    # Final validation and cleanup
    # Ensure no negative radii due to numerical noise
    best_radii = np.maximum(best_radii, 0)
    
    # Center clamping for safety (though constraints should handle it)
    best_centers[:, 0] = np.clip(best_centers[:, 0], best_radii, 1 - best_radii)
    best_centers[:, 1] = np.clip(best_centers[:, 1], best_radii, 1 - best_radii)

    return best_centers, best_radii, float(best_sum_radii)

# To verify locally
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Max radius: {np.max(radii)}, Min radius: {np.min(radii)}")
