# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=1ed4a012 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # 1. Generate a dense hexagonal initialization
    # We create 5 rows. Alternating counts of 5 and 4 help distribute 26 circles efficiently.
    initial_centers = []
    # Row 0: 5 circles
    for i in range(5):
        initial_centers.append([0.1 + i * 0.2, 0.1])
    # Row 1: 5 circles
    for i in range(5):
        initial_centers.append([0.1 + i * 0.2, 0.3])
    # Row 2: 5 circles
    for i in range(5):
        initial_centers.append([0.1 + i * 0.2, 0.5])
    # Row 3: 5 circles
    for i in range(5):
        initial_centers.append([0.1 + i * 0.2, 0.7])
    # Row 4: 6 circles (to reach 26)
    for i in range(6):
        initial_centers.append([0.0833333 + i * 0.15, 0.9])

    initial_radii = np.full(n, 0.05)

    def objective(vars, weight):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i] = [vars[3 * i], vars[3 * i + 1]]
            radii[i] = vars[3 * i + 2]

        # Primary objective: maximize sum of radii
        obj = -np.sum(radii)
        
        penalty = 0.0
        
        # 1. Boundary violations
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        viol_left = np.maximum(0, radii - centers[:, 0])
        viol_right = np.maximum(0, centers[:, 0] + radii - 1.0)
        viol_bottom = np.maximum(0, radii - centers[:, 1])
        viol_top = np.maximum(0, centers[:, 1] + radii - 1.0)
        boundary_viol = viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2
        penalty += weight * np.sum(boundary_viol)
        
        # 2. Pairwise overlaps
        # d >= r_i + r_j => violation if r_i + r_j - d > 0
        radii_sum = radii[:, None] + radii[None, :]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        overlaps = radii_sum - dists
        np.fill_diagonal(overlaps, 0) # Ignore self-overlap
        pair_viol = np.maximum(0, overlaps)**2
        penalty += weight * np.sum(pair_viol)
        
        return obj + penalty

    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    initial_vars = np.zeros(n * 3)
    for i in range(n):
        initial_vars[3 * i] = initial_centers[i][0]
        initial_vars[3 * i + 1] = initial_centers[i][1]
        initial_vars[3 * i + 2] = initial_radii[i]

    # Optimization runs with different penalty weights and slight perturbations
    for w in [1000.0, 5000.0]:
        for seed in range(5):
            np.random.seed(seed)
            vars_perturbed = initial_vars + np.random.normal(0, 0.005, n * 3)
            
            # Clamp perturbed vars to bounds
            for k in range(0, len(vars_perturbed), 3):
                vars_perturbed[k] = np.clip(vars_perturbed[k], 0, 1)
                vars_perturbed[k+1] = np.clip(vars_perturbed[k+1], 0, 1)
                vars_perturbed[k+2] = np.clip(vars_perturbed[k+2], 0, 0.5)

            try:
                res = minimize(objective, vars_perturbed, method='L-BFGS-B', 
                               args=(w,), bounds=bounds, 
                               options={'maxiter': 3000, 'ftol': 1e-12})
                
                current_centers = np.zeros((n, 2))
                current_radii = np.zeros(n)
                for i in range(n):
                    current_centers[i] = [res.x[3 * i], res.x[3 * i + 1]]
                    current_radii[i] = res.x[3 * i + 2]
                
                # Check validity roughly (penalty should be near 0)
                penalty_val = res.fun + np.sum(current_radii)
                if penalty_val < 1e-4: 
                    current_sum = np.sum(current_radii)
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = current_centers.copy()
                        best_radii = current_radii.copy()
            except Exception:
                continue

    # If optimization failed to find a valid result, fallback to initial
    if best_centers is None:
        return np.array(initial_centers), initial_radii, np.sum(initial_radii)

    return best_centers, best_radii, best_sum_radii
