# sol_000301 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2823a898) state=3b550971 sum of radii=2.614398 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Number of restarts to find a robust global optimum
    n_restarts = 20

    def objective(vars):
        radii = vars[2*n:]
        return -np.sum(radii)

    def constraints_func(vars):
        centers = vars[:2*n].reshape(n, 2)
        radii = vars[2*n:]
        
        vals = []
        # Wall constraints: x - r >= 0, 1 - x - r >= 0, etc.
        vals.append(centers[:, 0] - radii)
        vals.append(1 - centers[:, 0] - radii)
        vals.append(centers[:, 1] - radii)
        vals.append(1 - centers[:, 1] - radii)
        
        # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        diff = dist_sq - r_sum_sq
        # Extract upper triangle to avoid duplicate constraints (i, j) and (j, i)
        idx = np.triu_indices(n, k=1)
        vals.append(diff[idx])
        
        return np.concatenate(vals)

    cons = {'type': 'ineq', 'fun': constraints_func}
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n # Radii bounded by 0.5

    for _ in range(n_restarts):
        centers_init = np.random.rand(n, 2)
        radii_init = np.full(n, 0.02)
        x0 = np.concatenate([centers_init.flatten(), radii_init])

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
            
            if res.success:
                current_centers = res.x[:2*n].reshape(n, 2)
                current_radii = res.x[2*n:]
                current_sum = np.sum(current_radii)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = current_centers
                    best_radii = current_radii
        except Exception:
            pass

    # Try a structured grid initialization as well
    for _ in range(5):
        # Hexagonal-like grid
        y_coords = np.linspace(0.1, 0.9, 6)
        x_coords = np.linspace(0.1, 0.9, 5)
        grid = np.array([[x, y] for y in y_coords for x in x_coords])
        np.random.shuffle(grid)
        
        centers_init = grid[:n]
        radii_init = np.full(n, 0.01)
        x0 = np.concatenate([centers_init.flatten(), radii_init])

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
            if res.success:
                current_sum = np.sum(res.x[2*n:])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x[:2*n].reshape(n, 2)
                    best_radii = res.x[2*n:]
        except Exception:
            pass
            
    return best_centers, best_radii, best_sum

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
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
