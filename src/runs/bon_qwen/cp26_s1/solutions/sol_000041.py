# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6f5bcc91) state=855155e6 sum of radii=1.380573 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initial Configuration: Dense Hexagonal Grid
    centers = np.zeros((n, 2))
    idx = 0
    # Generate points in a staggered grid pattern
    rows = 6
    cols = 5
    x_step = 1.0 / (cols + 0.5)
    y_step = 1.0 / (rows + 0.5)
    
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            offset = 0.5 * x_step if i % 2 == 1 else 0.0
            centers[idx] = [offset + (j + 0.5) * x_step, (i + 0.5) * y_step]
            idx += 1
        if idx >= n:
            break
            
    # Fill remaining if any (should be exact or close)
    while idx < n:
        centers[idx] = np.random.rand(2) * 0.6 + 0.2
        idx += 1
        
    # 2. Force-Directed Pre-optimization
    r_sim = 0.09
    dt = 0.008
    damping = 0.85
    
    for step in range(3000):
        forces = np.zeros_like(centers)
        max_viol = 0.0
        
        for i in range(n):
            xi, yi = centers[i]
            # Boundary forces
            if xi < r_sim: forces[i, 0] += (r_sim - xi) * 20.0
            elif xi > 1 - r_sim: forces[i, 0] -= (xi - (1 - r_sim)) * 20.0
            if yi < r_sim: forces[i, 1] += (r_sim - yi) * 20.0
            elif yi > 1 - r_sim: forces[i, 1] -= (yi - (1 - r_sim)) * 20.0
            
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = 2 * r_sim
                if dist < min_dist and dist > 1e-7:
                    # Repulsive force
                    f_mag = (min_dist - dist) * 15.0 / dist
                    forces[i] += f_mag * diff
                    forces[j] -= f_mag * diff
                    max_viol = max(max_viol, min_dist - dist)
        
        centers += dt * forces
        centers *= damping
        centers = np.clip(centers, 1e-6, 1 - 1e-6)
        
        # Adaptive radius growth during simulation
        if max_viol < 1e-4:
            r_sim += 2e-5
            
    # 3. Gradient-Based Refinement using Scipy
    def objective(vars_flat):
        r = vars_flat[-1]
        c = vars_flat[:-1].reshape(n, 2)
        loss = -r
        penalty = 0.0
        
        # Boundary penalty
        for i in range(n):
            x, y = c[i]
            if x < r: penalty += (r - x)**2
            if x > 1 - r: penalty += (x - (1 - r))**2
            if y < r: penalty += (r - y)**2
            if y > 1 - r: penalty += (y - (1 - r))**2
            
        # Pairwise distance penalty
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                if d < 2 * r:
                    penalty += (2 * r - d)**2
                    
        return loss + 5000.0 * penalty

    x0 = np.concatenate([centers.flatten(), [r_sim]])
    bounds = [(0.0, 1.0) for _ in range(2 * n)] + [(0.05, 0.5)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 3000, 'ftol': 1e-10})
    
    best_centers = res.x[:-1].reshape(n, 2)
    best_r = res.x[-1]
    
    # 4. Validation & Exact Radius Calculation
    # Compute the strictly feasible radius based on final positions
    min_gap = 1.0
    for i in range(n):
        x, y = best_centers[i]
        d_wall = min(x, 1 - x, y, 1 - y)
        if d_wall < min_gap:
            min_gap = d_wall
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            if d / 2 < min_gap:
                min_gap = d / 2
                
    # Use the tightest constraint to ensure validity
    final_r = min(best_r, min_gap)
    
    # Final clamp to ensure strict boundaries
    final_centers = np.clip(best_centers, final_r + 1e-9, 1 - final_r - 1e-9)
    final_radii = np.full(n, final_r)
    
    return final_centers, final_radii, np.sum(final_radii)
