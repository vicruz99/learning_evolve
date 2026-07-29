# sol_000334 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9bf69ab6) state=7e05feb5 sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26

    # --- Helper functions for optimization ---
    
    def objective(x):
        # Maximize sum of radii -> Minimize negative sum
        # Radii are stored at indices 3*i + 2
        radii_sum = 0.0
        for i in range(n):
            radii_sum += x[3 * i + 2]
        return -radii_sum

    def get_constraints(x):
        # Returns a list of constraint values for SLSQP
        # Inequalities: c(x) >= 0
        constraints = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        for i in range(n):
            xi = x[3 * i]
            yi = x[3 * i + 1]
            ri = x[3 * i + 2]
            
            constraints.append(xi - ri)       # x >= r
            constraints.append(1 - xi - ri)   # 1-x >= r
            constraints.append(yi - ri)       # y >= r
            constraints.append(1 - yi - ri)   # 1-y >= r

        # Non-overlap constraints: dist^2 >= (ri + rj)^2
        for i in range(n):
            xi = x[3 * i]
            yi = x[3 * i + 1]
            ri = x[3 * i + 2]
            
            for j in range(i + 1, n):
                xj = x[3 * j]
                yj = x[3 * j + 1]
                rj = x[3 * j + 2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum_sq = (ri + rj)**2
                constraints.append(dist_sq - rad_sum_sq)
                
        return np.array(constraints)

    # --- Initialization 1: Structured Hexagonal Packing ---
    # 6 rows with counts 5, 4, 5, 4, 5, 3
    row_counts = [5, 4, 5, 4, 5, 3]
    r_init = 0.09
    x0_1 = np.zeros(3 * n)
    
    current_idx = 0
    for row, count in enumerate(row_counts):
        y = r_init + row * np.sqrt(3) * r_init
        # Shift odd rows by r_init horizontally
        x_start = r_init + (r_init if row % 2 == 1 else 0)
        
        for col in range(count):
            x = x_start + col * 2 * r_init
            # Clamp to [0, 1] just in case, though math suggests it fits
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            
            x0_1[3 * current_idx] = x
            x0_1[3 * current_idx + 1] = y
            x0_1[3 * current_idx + 2] = r_init
            current_idx += 1

    # --- Initialization 2: Perturbed Hexagonal Packing ---
    x0_2 = x0_1.copy()
    # Add small random noise to centers and radii
    np.random.seed(42) # For reproducibility
    noise = np.random.normal(0, 0.01, 3 * n)
    # Keep radii positive and small
    noise[2::3] = np.abs(noise[2::3]) * 0.5
    x0_2 = np.clip(x0_2 + noise, 0.01, 0.99)
    # Ensure radii are not too large initially
    x0_2[2::3] = np.minimum(x0_2[2::3], 0.1)

    # --- Bounds ---
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # --- Optimization ---
    # We run optimization on both initializations and pick the best
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    for init_guess in [x0_1, x0_2]:
        # Define constraint for scipy
        cons = {'type': 'ineq', 'fun': get_constraints}
        
        # Run optimization
        # maxiter 2000 allows it to settle into tight packing
        res = minimize(
            objective, 
            init_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
        )

        # Extract results
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = res.x[3 * i]
            centers[i, 1] = res.x[3 * i + 1]
            radii[i] = res.x[3 * i + 2]
            
        current_sum = np.sum(radii)
        
        # Basic validation check (local)
        valid = True
        # Check boundaries
        for i in range(n):
            if (radii[i] < 1e-7 or centers[i,0] < radii[i] - 1e-9 or 
                centers[i,0] > 1 - radii[i] + 1e-9 or
                centers[i,1] < radii[i] - 1e-9 or 
                centers[i,1] > 1 - radii[i] + 1e-9):
                valid = False
                break
        
        # Check overlaps
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
        
        if valid and current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Fallback if optimization failed (should not happen with good init)
    if best_centers is None:
        best_centers = np.zeros((26, 2))
        best_radii = np.zeros(26)
        for i in range(26):
            best_centers[i] = [0.5, 0.5]
            best_radii[i] = 0.001
            
    return best_centers, best_radii, best_sum_radii
