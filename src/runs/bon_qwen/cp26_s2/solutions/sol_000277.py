# sol_000277 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9d8cea89) state=fd9df349 sum of radii=2.311903 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We arrange 26 points in a hexagonal pattern (6 rows of 5, with the last one adjusted).
    # This pattern is denser than a square grid.
    centers = np.zeros((n, 2))
    
    # Define the hexagonal grid coordinates before scaling
    # Rows 0 to 5
    row_counts = [5, 5, 5, 5, 5, 1]
    
    idx = 0
    for r in range(6):
        count = row_counts[r]
        y_base = r * (np.sqrt(3) / 2.0)
        
        for c in range(count):
            if r % 2 == 0:
                x_coord = c
            else:
                x_coord = c + 0.5
            
            centers[idx, 0] = x_coord
            centers[idx, 1] = y_base
            idx += 1
            
    # Scale and center the grid to fit roughly in [0, 1] x [0, 1]
    # Find min and max of x and y
    min_x, max_x = centers[:, 0].min(), centers[:, 0].max()
    min_y, max_y = centers[:, 1].min(), centers[:, 1].max()
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Scaling factor to fit within unit square with some margin
    # We want to map [min, max] to [margin, 1-margin]
    margin = 0.1
    scale_x = (1 - 2 * margin) / width
    scale_y = (1 - 2 * margin) / height
    scale = min(scale_x, scale_y)
    
    # Apply scaling and centering
    centers[:, 0] = (centers[:, 0] - min_x) * scale + margin
    centers[:, 1] = (centers[:, 1] - min_y) * scale + margin
    
    # 2. Optimization: Maximize Minimum Distance
    # We use Nelder-Mead to maximize the minimum distance between centers and boundaries.
    # This is equivalent to maximizing the radius of equal circles.
    
    def objective(x):
        """
        Objective function to minimize.
        We minimize the negative of the minimum distance (dist_min).
        """
        c = x.reshape((n, 2))
        
        # 1. Distance to boundaries
        # For a circle at (x, y) with radius r, it must be inside [r, 1-r]
        # So distance to boundary is min(x, 1-x, y, 1-y)
        # We want to maximize this distance.
        d_bound = np.min(np.array([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]]).T, axis=1)
        
        # 2. Distance between centers
        # For circles i and j, dist >= 2r.
        # We consider dist(c_i, c_j) / 2 as the candidate radius.
        # We want to maximize min(dist(c_i, c_j) / 2).
        
        # Compute all pairwise distances
        # Using broadcasting for efficiency
        # diff shape (n, n, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # We only care about i < j, but min over all off-diagonals is same
        # Mask out diagonal
        np.fill_diagonal(dists, np.inf)
        d_centers = np.min(dists) / 2.0
        
        # The limiting radius is the minimum of boundary and center distances
        min_r = min(np.min(d_bound), d_centers)
        
        # We want to maximize min_r, so we return -min_r
        return -min_r

    # Run optimization
    # Nelder-Mead is robust for non-smooth functions like min()
    result = minimize(objective, centers.flatten(), method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_centers = result.x.reshape((n, 2))
    
    # 3. Calculate Final Radii
    # Compute the tightest constraint (radius) for the optimized centers
    c = best_centers
    
    # Boundary constraints
    r_bound = np.min(np.array([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]]).T, axis=1)
    r_boundary_limit = np.min(r_bound)
    
    # Pairwise constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pairwise_limit = np.min(dists) / 2.0
    
    # The valid radius is the minimum of these limits
    final_radius = min(r_boundary_limit, r_pairwise_limit)
    
    # Ensure non-negative
    final_radius = max(0.0, final_radius)
    
    radii = np.full(n, final_radius)
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii

# Helper to run and validate locally if needed (not part of the required function structure, but good for testing)
if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    # print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")
    # print(f"Centers:\n{centers}")
