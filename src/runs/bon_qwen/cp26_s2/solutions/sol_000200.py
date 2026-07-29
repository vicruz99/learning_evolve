# sol_000200 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=5caf9846 sum of radii=1.282505 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    def hex_init(angle=0.0):
        r_est = 0.095
        centers = []
        rows = 6
        for i in range(rows):
            row_r = r_est
            y = row_r + i * np.sqrt(3) * row_r
            num_cols = int(np.floor((1.0 - 2 * row_r) / (2 * row_r)) + 1)
            if (i % 2) == 1:
                num_cols = max(0, num_cols - 1)
            
            for j in range(num_cols):
                if len(centers) >= n:
                    break
                x = row_r + j * 2 * row_r
                cx, cy = x - 0.5, y - 0.5
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                centers.append([cx * cos_a - cy * sin_a + 0.5, cx * sin_a + cy * cos_a + 0.5])
        
        centers = np.array(centers[:n])
        radii = np.full(n, r_est)
        return centers, radii

    def objective(vars):
        centers = vars[:n * 2].reshape(n, 2)
        radii = vars[n * 2:]
        loss = 0.0
        
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            # Boundary penalty
            if x - r < 0: loss += (x - r) ** 2
            if x + r > 1: loss += (x + r - 1) ** 2
            if y - r < 0: loss += (y - r) ** 2
            if y + r > 1: loss += (y + r - 1) ** 2
            
            # Overlap penalty
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    loss += overlap ** 2
        return loss

    for ang in [0.0, 0.046, 0.08, -0.05, 0.15]:
        c, r = hex_init(ang)
        initial = np.concatenate([c.flatten(), r])
        
        res = minimize(
            objective, 
            initial, 
            method='L-BFGS-B', 
            bounds=[(0, 1)] * (n * 2) + [(0, 0.25)] * n,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        c_opt = res.x[:n * 2].reshape(n, 2)
        r_opt = res.x[n * 2:]
        
        # Final scaling for maximum valid radius
        min_dist = np.inf
        for i in range(n):
            r_opt[i] = min(c_opt[i, 0], 1 - c_opt[i, 0], c_opt[i, 1], 1 - c_opt[i, 1])
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((c_opt[i] - c_opt[j]) ** 2))
                    min_dist = min(min_dist, d - r_opt[j])
            r_opt[i] = min(r_opt[i], max(0, min_dist / 2))

        if np.sum(r_opt) > best_sum:
            best_sum = np.sum(r_opt)
            best_centers = c_opt
            best_radii = r_opt

    return best_centers, best_radii, best_sum
