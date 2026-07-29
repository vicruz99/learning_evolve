# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 26e3ad40) state=0c0fc664 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal-like arrangement
    # Distribute 26 circles into rows to balance dimensions
    # Counts: 5, 4, 5, 4, 5, 3 (Total = 26)
    row_counts = [5, 4, 5, 4, 5, 3]
    initial_r = 0.10
    centers = []
    
    y = initial_r
    for i, count in enumerate(row_counts):
        row_centers = []
        # Horizontal spacing for hexagonal packing
        spacing = 2 * initial_r
        # Offset every other row to nestle circles
        offset = initial_r if i % 2 == 1 else 0
        
        # Center the row horizontally within [0, 1]
        # Width of row = (count - 1) * spacing + 2 * initial_r
        row_width = (count - 1) * spacing + 2 * initial_r
        start_x = (1.0 - row_width) / 2.0 + offset
        
        for j in range(count):
            x = start_x + j * spacing
            centers.append([x, y])
        
        # Vertical spacing for hexagonal packing
        y += np.sqrt(3) * initial_r

    centers = np.array(centers)

    # 2. Optimization: Refine positions to maximize the minimum radius
    # We optimize centers to maximize the minimum clearance to boundaries and other circles.
    # This is equivalent to finding the best layout for a fixed 'large' radius.
    
    def objective(x_flat):
        # We want to minimize the negative of the "bottleneck radius".
        # However, for optimization stability, we can just minimize a potential function
        # or maximize the minimum distance. 
        # Here we define a function that returns the negative of the max possible radius r
        # for the current configuration of centers.
        
        c = x_flat.reshape(-1, 2)
        min_dist = 1.0 # Upper bound
        
        # Boundary constraints: dist to edge must be >= r
        # x - r >= 0 => r <= x
        # 1 - x - r >= 0 => r <= 1 - x
        # Same for y
        min_dist = min(min_dist, np.min(c[:, 0]))
        min_dist = min(min_dist, np.min(1.0 - c[:, 0]))
        min_dist = min(min_dist, np.min(c[:, 1]))
        min_dist = min(min_dist, np.min(1.0 - c[:, 1]))
        
        # Inter-circle constraints: dist(c_i, c_j) >= 2r => r <= dist/2
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                if dist < min_dist * 2:
                    min_dist = dist / 2.0
                    
        return -min_dist # Minimize negative max radius

    # Initial bounds for centers: [0, 1] for x and y
    bounds = [(0, 1)] * (2 * n_circles)
    
    # Use L-BFGS-B for bound constrained optimization
    result = minimize(objective, centers.flatten(), method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    optimized_centers = result.x.reshape(-1, 2)
    
    # 3. Calculate the actual maximum radius for the optimized layout
    # Re-evaluate the bottleneck radius precisely
    max_r = 1.0
    
    # Check boundaries
    max_r = min(max_r, np.min(optimized_centers[:, 0]))
    max_r = min(max_r, np.min(1.0 - optimized_centers[:, 0]))
    max_r = min(max_r, np.min(optimized_centers[:, 1]))
    max_r = min(max_r, np.min(1.0 - optimized_centers[:, 1]))
    
    # Check distances
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((optimized_centers[i] - optimized_centers[j])**2))
            max_r = min(max_r, dist / 2.0)
            
    # Finalize radii
    radii = np.full(n_circles, max_r)
    sum_radii = np.sum(radii)
    
    return optimized_centers, radii, sum_radii

# Note: The validation function is not included here as per instructions, 
# but the output format matches the requirements.
