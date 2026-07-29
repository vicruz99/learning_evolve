# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000044 (state 69bc282d) state=672c6201 sum of radii=2.393821 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 26

def get_max_radii(centers):
    """
    Computes the maximum valid radius for each circle given fixed centers.
    r_i = min(dist to boundary, 0.5 * min(dist to other centers))
    """
    n = centers.shape[0]
    # Distance to boundaries
    rb = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(rb, rp)

def neg_sum_radii(x):
    """Objective function for optimizers: minimize negative sum of radii."""
    centers = x.reshape(N, 2)
    return -np.sum(get_max_radii(centers))

def run_packing() -> tuple:
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Global search using Differential Evolution
    # DE effectively navigates the rugged, non-smooth objective landscape
    res_de = differential_evolution(
        neg_sum_radii, bounds, seed=42, maxiter=300, 
        popsize=10, tol=1e-8, mutation=(0.5, 1.2), 
        recombination=0.8
    )
    best_centers = res_de.x.reshape(N, 2)
    
    # Phase 2: Local refinement using Nelder-Mead
    # Handles non-smooth minima and fine-tunes positions
    for _ in range(5):
        res_nm = minimize(
            neg_sum_radii, best_centers.flatten(), method='Nelder-Mead',
            options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        best_centers = res_nm.x.reshape(N, 2)
        
    # Phase 3: Coordinate ascent local search
    # Directly perturbs each center to maximize the sum of radii.
    # This greedy phase resolves tight local bottlenecks that continuous 
    # optimizers often miss due to the piecewise-linear nature of the objective.
    step = 0.015
    for iteration in range(8):
        improved = True
        while improved:
            improved = False
            curr_r = get_max_radii(best_centers)
            curr_sum = np.sum(curr_r)
            
            for i in range(N):
                best_move = best_centers[i].copy()
                best_s = curr_sum
                
                # Try multiple random directions around the current position
                for _ in range(10):
                    d = np.random.randn(2)
                    d /= np.linalg.norm(d)
                    new_pos = best_centers[i] + d * step
                    
                    if 0 <= new_pos[0] <= 1 and 0 <= new_pos[1] <= 1:
                        best_centers[i] = new_pos
                        new_sum = np.sum(get_max_radii(best_centers))
                        if new_sum > best_s:
                            best_s = new_sum
                            best_move = new_pos.copy()
                            
                best_centers[i] = best_move
                if best_s > curr_sum + 1e-9:
                    curr_sum = best_s
                    improved = True
                    
        step *= 0.65
        
    final_radii = get_max_radii(best_centers)
    return best_centers, final_radii, float(np.sum(final_radii))
