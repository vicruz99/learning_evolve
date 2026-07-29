# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eea877f1) state=131796c4 sum of radii=2.323951 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform

# Number of circles
N_CIRCLES = 26

def compute_radii_and_sum(centers):
    """
    Computes the maximal radii for given centers and returns radii array and sum.
    Centers shape: (N, 2)
    """
    # Compute pairwise distances
    # pdist computes distances between all pairs, returns 1D array
    dists_flat = pdist(centers, 'euclidean')
    
    # Convert to square form (N x N)
    dists_matrix = squareform(dists_flat)
    
    # Set diagonal to infinity so min doesn't pick 0
    np.fill_diagonal(dists_matrix, np.inf)
    
    # Minimum distance to any other circle for each circle
    min_neighbor_dists = np.min(dists_matrix, axis=1)
    
    # Radius constrained by neighbors: half the distance to the closest neighbor
    r_from_neighbors = min_neighbor_dists / 2.0
    
    # Radius constrained by boundaries
    x = centers[:, 0]
    y = centers[:, 1]
    # Distance to 4 walls
    r_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # The actual radius is the minimum of these constraints
    radii = np.minimum(r_from_neighbors, r_boundary)
    
    # Ensure non-negative (though math guarantees it)
    radii = np.maximum(radii, 0.0)
    
    return radii, np.sum(radii)

def objective_function(vars_1d):
    """
    Objective function for optimization. Minimizes negative sum of radii.
    """
    centers = vars_1d.reshape(-1, 2)
    _, sum_radii = compute_radii_and_sum(centers)
    return -sum_radii

def local_optimization(centers, iterations=2000):
    """
    Perform a local hill-climbing search to refine center positions.
    """
    current_centers = centers.copy()
    N = current_centers.shape[0]
    
    # Compute initial sum
    _, current_sum = compute_radii_and_sum(current_centers)
    
    # Initial step size
    step_size = 0.05
    
    for i in range(iterations):
        # Adaptive step size decay could be added, but fixed is simple
        # Try to improve by moving a random circle
        idx = np.random.randint(N)
        
        # Random direction
        delta = np.random.uniform(-1, 1, 2)
        delta = delta / np.linalg.norm(delta) * step_size
        
        new_center = current_centers[idx] + delta
        
        # Boundary clipping
        new_center = np.clip(new_center, 1e-6, 1.0 - 1e-6)
        
        # Test move
        old_pos = current_centers[idx].copy()
        current_centers[idx] = new_center
        
        # We only need to recompute radii for this configuration.
        # Optimization: only recompute affected parts? 
        # For N=26, full recompute is fast enough.
        _, new_sum = compute_radii_and_sum(current_centers)
        
        if new_sum > current_sum:
            current_sum = new_sum
            # If we found an improvement, maybe keep step size or reduce slightly
            step_size = np.clip(step_size * 1.01, 0.001, 0.1) 
        else:
            # Revert
            current_centers[idx] = old_pos
            # Reduce step size if no improvement found often?
            # Simple decay
            if i % 100 == 0 and i > 0:
                step_size *= 0.9

    return current_centers

def run_packing():
    """
    Main function to run the packing optimization.
    Returns (centers, radii, sum_radii).
    """
    # Bounds for each coordinate: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES)
    
    # Phase 1: Global Optimization using Differential Evolution
    # popsize=5 means 5 * 52 = 260 individuals. 
    # maxiter=100. This should be fast enough.
    # seed for reproducibility
    try:
        result = differential_evolution(
            objective_function, 
            bounds, 
            popsize=5, 
            maxiter=100, 
            tol=1e-7, 
            seed=42,
            mutation=(0.5, 1.5),
            recombination=0.9
        )
        best_centers = result.x.reshape(-1, 2)
    except Exception:
        # Fallback if scipy fails or is too slow (unlikely)
        # Random initialization
        best_centers = np.random.rand(N_CIRCLES, 2)
        
    # Phase 2: Local Optimization
    # Refine the solution found by DE
    refined_centers = local_optimization(best_centers, iterations=3000)
    
    # Compute final radii and sum
    final_radii, final_sum = compute_radii_and_sum(refined_centers)
    
    # Return in required format
    # centers: (26, 2), radii: (26,), sum_radii: float
    return refined_centers, final_radii, final_sum

# Validation check (optional, but good for sanity)
# This block is not executed during submission usually, but helps in thought process
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)
