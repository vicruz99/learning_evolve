# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=6e04cff5 sum of radii=1.698614 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

def compute_radii(centers):
    """
    Computes the maximum possible radii for given centers such that circles
    do not overlap and are inside the unit square.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates.
        
    Returns:
        radii: np.array of shape (n) with radius of each circle.
    """
    n = centers.shape[0]
    
    # 1. Distance to boundaries
    # For a circle at (x, y) with radius r to be inside [0,1]^2:
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    # So r <= min(x, 1-x, y, 1-y)
    dist_to_left = centers[:, 0]
    dist_to_right = 1.0 - centers[:, 0]
    dist_to_bottom = centers[:, 1]
    dist_to_top = 1.0 - centers[:, 1]
    
    wall_distances = np.minimum.reduce([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top])
    
    # 2. Distance to other circles
    # r_i + r_j <= distance(i, j)
    # To maximize sum of radii, we generally want r_i approx r_j, 
    # so r_i <= distance(i, j) / 2 is a good local constraint if we consider 
    # the radius limited by the closest neighbor.
    # Specifically, if we fix centers, the max radius for circle i is limited 
    # by half the distance to its nearest neighbor (assuming neighbors have similar radii).
    # More precisely, r_i <= min_{j!=i} (dist(i,j) - r_j). 
    # However, for a static calculation given fixed centers, the max radius r_i 
    # such that it doesn't overlap with ANY other circle j (with radius r_j) 
    # is complex because r_j depends on r_i.
    # But a valid upper bound for r_i is 0.5 * min_{j!=i} dist(i, j).
    # If all circles have the same radius r, then r = 0.5 * min_dist.
    # For varying radii, this is a safe conservative estimate that ensures non-overlap 
    # if we assume the "neighbor" also expands to fill the gap? 
    # Actually, strictly speaking, if we just want valid radii, 
    # we can set r_i = min(wall_dist_i, 0.5 * min_neighbor_dist_i).
    # This ensures r_i + r_j <= 0.5 d_ij + 0.5 d_ij = d_ij ? 
    # No. r_i <= 0.5 d_ik, r_k <= 0.5 d_ik. So r_i + r_k <= d_ik. Correct.
    # This is a valid packing construction.
    
    # Compute pairwise distances
    # cdist is efficient
    dist_matrix = cdist(centers, centers)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find minimum distance to any other circle for each circle
    min_neighbor_distances = np.min(dist_matrix, axis=1)
    
    # Radius is limited by half the distance to the nearest neighbor
    # and by the distance to the walls.
    radii = np.minimum(wall_distances, 0.5 * min_neighbor_distances)
    
    return radii

def objective(centers_flat):
    """
    Objective function for optimization.
    Minimizes the negative sum of radii (maximizing sum of radii).
    """
    centers = centers_flat.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    # Bounds for centers: x in [0, 1], y in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Use Differential Evolution for global optimization
    # It handles non-smooth objective functions well.
    # We use a seed for reproducibility, but the algorithm is stochastic.
    # popsize and maxiter tuned for balance between speed and quality.
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,       # Number of iterations
            popsize=20,        # Population size
            mutation=(0.5, 1.5), # Mutation constant
            recombination=0.7, # Crossover probability
            seed=42,           # Random seed
            polish=False       # Disable local polishing as objective is non-smooth
        )
        
        best_centers_flat = result.x
        best_centers = best_centers_flat.reshape(n, 2)
        
        # Compute final radii based on the optimized centers
        best_radii = compute_radii(best_centers)
        total_sum_radii = np.sum(best_radii)
        
        # Ensure non-negative radii (should be guaranteed by logic, but safe check)
        best_radii = np.maximum(best_radii, 0.0)
        
        return best_centers, best_radii, total_sum_radii
        
    except Exception as e:
        # Fallback to a simple grid packing if optimization fails
        # Although unlikely, good for robustness.
        # 5x5 grid has 25 circles. We need 26.
        # Let's just return a valid grid packing for 25 circles and one tiny one?
        # Or just 26 circles in a grid pattern.
        # A simple heuristic fallback:
        centers_fallback = np.zeros((n, 2))
        # Try to fit 26 in a grid
        # 5 rows, roughly 5 per row.
        # Just space them out.
        step_x = 1.0 / 6
        step_y = 1.0 / 5
        
        idx = 0
        for r in range(5):
            for c in range(6): # 30 slots, we take 26
                if idx < n:
                    centers_fallback[idx, 0] = (c + 0.5) * step_x
                    centers_fallback[idx, 1] = (r + 0.5) * step_y
                    idx += 1
        
        radii_fallback = compute_radii(centers_fallback)
        return centers_fallback, radii_fallback, np.sum(radii_fallback)

# To verify the solution locally (optional, not part of submission output)
if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Total sum of radii: {total:.5f}")
    print(f"Min radius: {np.min(radii):.5f}, Max radius: {np.max(radii):.5f}")
