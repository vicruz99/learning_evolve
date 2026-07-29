# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=34d30162 sum of radii=2.496579 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = 0.0

    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        radii = vars[2::3]
        return -np.sum(radii)  # Minimize negative sum

    def constraint_boundary(vars):
        centers = np.array([vars[i::3] for i in range(2)]).T  # Shape (n, 2)
        radii = vars[2::3]
        constraints = []
        for i in range(n):
            # x - r >= 0  => x - r >= 0
            constraints.append(centers[i, 0] - radii[i])
            # x + r <= 1  => 1 - x - r >= 0
            constraints.append(1.0 - centers[i, 0] - radii[i])
            # y - r >= 0
            constraints.append(centers[i, 1] - radii[i])
            # y + r <= 1
            constraints.append(1.0 - centers[i, 1] - radii[i])
            # r >= 0 (handled by bounds, but good to have slack)
            constraints.append(radii[i]) 
        return np.array(constraints)

    def constraint_overlap(vars):
        centers = np.array([vars[i::3] for i in range(2)]).T
        radii = vars[2::3]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # dist >= ri + rj  => dist - ri - rj >= 0
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)

    # Helper to create hexagonal initialization
    def get_hex_init(r_offset=0.05):
        # Try to fit roughly 26 circles in hexagonal pattern
        # Rows of 5 and 6?
        # 6, 5, 6, 5, 4 -> 26 circles
        # Or just random dense packing
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.09) # Initial guess
        
        # Hexagonal grid points
        points = []
        y = 0.1 + 0.05 # Start slightly inside
        row_len = 0
        r_est = 0.1
        dy = np.sqrt(3) * r_est
        
        # Generate enough points
        while len(points) < n + 10:
            x = 0.1
            if len(points) % 2 == 1:
                x = 0.1 + r_est # Stagger
            while x < 0.9:
                points.append([x, y])
                x += 2 * r_est
            y += dy
        
        # Select n points closest to center or just first n
        # Centering the selection
        points = np.array(points[:n])
        
        # Center and scale to fit better
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        # Simple scaling to [0.1, 0.9] range
        if x_max > x_min:
            points[:, 0] = 0.1 + (points[:, 0] - x_min) / (x_max - x_min) * 0.8
        if y_max > y_min:
            points[:, 1] = 0.1 + (points[:, 1] - y_min) / (y_max - y_min) * 0.8
            
        centers = points
        return centers, radii

    # Helper to create random initialization
    def get_random_init():
        # Poisson disk sampling approx or just random jitter
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        radii = np.full(n, 0.08)
        return centers, radii

    # Bounds for variables [x, y, r]
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    # Run optimization multiple times with different seeds/inits
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        
        # Mix of initializations
        if seed < 2:
            centers, radii = get_hex_init()
        else:
            centers, radii = get_random_init()

        # Flatten variables
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.array([res.x[i::3] for i in range(2)]).T
                    best_radii = res.x[2::3]
        except Exception:
            continue

    # Final validation check and clipping if necessary due to numerical noise
    # Ensure strict validity for the checker
    # If any circle is slightly outside, clamp it and reduce radius
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        
        # Clamp center
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Adjust radius to fit boundary
        r = min(r, x, 1-x, y, 1-y)
        r = max(r, 0)
        
        best_centers[i] = (x, y)
        best_radii[i] = r

    # Recalculate sum
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
