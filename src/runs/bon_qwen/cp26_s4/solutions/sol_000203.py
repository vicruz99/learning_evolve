# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=ff24b188 sum of radii=2.594736 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    # 1. Initialize a 5x5 grid for 25 circles
    n = 26
    r_grid = 0.1
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Create 5x5 grid
    coords = []
    for i in range(5):
        for j in range(5):
            coords.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    centers[:25] = coords
    radii[:25] = r_grid

    # Add 26th circle in a gap (center of a square cell)
    # Gap at (0.2, 0.2) relative to bottom-left of cell? 
    # Grid points: (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3)
    # Center of gap: (0.2, 0.2)
    centers[25] = [0.2, 0.2]
    # Radius slightly smaller than gap size ~0.041
    radii[25] = 0.035

    # 2. Optimization
    # We optimize both positions and radii. 
    # However, to ensure a valid packing, we can use a force-based repulsion method
    # which is robust and doesn't require complex constraint handling.

    def objective(x):
        # x contains centers and radii
        c = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        
        # Negative sum of radii (to maximize)
        score = -np.sum(r)
        
        # Penalty for overlap
        overlap_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                r_sum = r[i] + r[j]
                if dist < r_sum:
                    overlap_penalty += (r_sum - dist)**2
        
        # Penalty for being outside square
        boundary_penalty = 0
        for i in range(n):
            x_c, y_c = c[i]
            r_c = r[i]
            if x_c - r_c < 0: boundary_penalty += (r_c - x_c)**2
            if x_c + r_c > 1: boundary_penalty += (x_c + r_c - 1)**2
            if y_c - r_c < 0: boundary_penalty += (r_c - y_c)**2
            if y_c + r_c > 1: boundary_penalty += (y_c + r_c - 1)**2
            
        return score + 1000 * overlap_penalty + 1000 * boundary_penalty

    # Initial state
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Optimize
    # Using a method that handles local landscapes well
    res = minimize(objective, x0, method='BFGS', options={'maxiter': 5000, 'ftol': 1e-12})
    
    best_c = res.x[:2*n].reshape(n, 2)
    best_r = res.x[2*n:]
    
    # Ensure non-negative radii
    best_r = np.maximum(best_r, 1e-6)
    
    # Final validation and adjustment to ensure strict feasibility
    # We might need to slightly shrink radii if numerical errors caused overlap
    for _ in range(100):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_c[i] - best_c[j])**2))
                r_sum = best_r[i] + best_r[j]
                if dist < r_sum:
                    # Shrink both slightly
                    shrink = (r_sum - dist) / 2 + 1e-7
                    best_r[i] -= shrink / 2
                    best_r[j] -= shrink / 2
                    valid = False
            # Boundary check
            for k in range(2):
                if best_c[i, k] - best_r[i] < 0:
                    best_r[i] = best_c[i, k]
                if best_c[i, k] + best_r[i] > 1:
                    best_r[i] = 1 - best_c[i, k]
        
        if valid:
            break

    sum_radii = np.sum(best_r)
    
    return best_c, best_r, sum_radii
