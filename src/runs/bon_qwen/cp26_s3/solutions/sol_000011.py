# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4ccca180) state=ac576690 sum of radii=1.375070 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def get_max_radius_equal(centers):
    """
    Calculates the maximum possible equal radius R for a given set of centers
    such that all circles fit in the unit square and do not overlap.
    """
    n = centers.shape[0]
    min_dist_to_boundary = float('inf')
    min_dist_between = float('inf')

    # 1. Check distance to boundaries
    # Constraint: x >= R, 1-x >= R => R <= min(x, 1-x)
    # Same for y. So R <= min(x, 1-x, y, 1-y)
    # We want the minimum over all circles.
    
    # Efficiently compute distances to boundaries
    # x coords: min(centers[:, 0], 1 - centers[:, 0])
    # y coords: min(centers[:, 1], 1 - centers[:, 1])
    
    dist_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    dist_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    
    min_boundary = np.min(np.minimum(dist_x, dist_y))
    min_dist_to_boundary = min_boundary

    # 2. Check distance between circles
    # Constraint: dist(ci, cj) >= 2*R => R <= dist(ci, cj) / 2
    
    # Compute pairwise distances
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    # sq_diff shape (n, n)
    
    # Optimized pairwise distance calculation
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    
    # Upper triangle (exclude diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sq_dist = sq_dist[mask]
    
    if len(sq_dist) > 0:
        min_dist_between = np.sqrt(np.min(sq_dist)) / 2.0
    else:
        min_dist_between = float('inf')

    # The radius is limited by the tightest constraint
    return min(min_dist_to_boundary, min_dist_between)

def optimize_packing():
    """
    Performs randomized local search to find optimal centers for 26 circles.
    """
    n_circles = 26
    best_centers = None
    best_r = 0.0
    
    # Number of restarts and iterations per restart
    n_restarts = 50
    n_iterations = 200
    step_size = 0.05
    
    for restart in range(n_restarts):
        # --- Initialization: Hexagonal Grid Perturbation ---
        # Generate a hexagonal grid
        # We want to cover the [0,1]x[0,1] square.
        # Generate points in a range slightly larger than square to ensure coverage
        centers = []
        
        # Hexagonal lattice parameters
        # Rows offset by half spacing
        spacing_x = 0.2 # Rough estimate, will be scaled/shifted
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Generate a dense cloud of hexagonal points
        x_start, x_end = -0.1, 1.1
        y_start, y_end = -0.1, 1.1
        
        x_vals = np.arange(x_start, x_end, spacing_x)
        y_vals = np.arange(y_start, y_end, spacing_y)
        
        grid_points = []
        for i, x in enumerate(x_vals):
            row_offset = (spacing_x / 2) * (i % 2)
            for y in y_vals:
                grid_points.append([x + row_offset, y])
        
        grid_points = np.array(grid_points)
        
        # Select n_circles points from grid that are best spaced
        # Heuristic: Pick points furthest from others or just random subset if grid is dense
        # Since grid is dense, let's just pick a random subset to vary restarts
        if len(grid_points) >= n_circles:
            indices = np.random.choice(len(grid_points), n_circles, replace=False)
            current_centers = grid_points[indices].copy()
        else:
            # Fallback to random
            current_centers = np.random.rand(n_circles, 2)
            
        # Clamp centers to be safely inside initially (e.g., [0.1, 0.9])
        # Actually, let's allow them to be anywhere but the radius calc handles boundaries.
        # Just ensure they are within [0,1] for validity.
        current_centers = np.clip(current_centers, 0.0, 1.0)
        
        # Initial optimization step
        current_r = get_max_radius_equal(current_centers)
        
        # Local Search
        for _ in range(n_iterations):
            # Try to move each circle randomly
            improved = False
            # Sort indices to process randomly
            perm = np.random.permutation(n_circles)
            
            for idx in perm:
                # Perturb circle idx
                original_pos = current_centers[idx].copy()
                
                # Random displacement
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                new_pos = original_pos + np.array([dx, dy])
                
                # Keep within bounds [0, 1]
                new_pos = np.clip(new_pos, 0.0, 1.0)
                
                # Check if valid move (inside square) - clip ensures this, but let's be safe
                # Actually clip might move it, which is a valid move in search space?
                # Better to reject if it goes out, or just clamp. Clamping is fine.
                
                # Evaluate new radius
                temp_centers = current_centers.copy()
                temp_centers[idx] = new_pos
                new_r = get_max_radius_equal(temp_centers)
                
                if new_r > current_r:
                    current_centers = temp_centers
                    current_r = new_r
                    improved = True
            
            # If no improvement, reduce step size slightly or keep going
            # If stuck, maybe a larger jump?
            if not improved:
                # Random jump for one circle
                idx = np.random.randint(n_circles)
                current_centers[idx] = np.random.rand(2) # Reset to random
                current_r = get_max_radius_equal(current_centers)

        # Update global best
        if current_r > best_r:
            best_r = current_r
            best_centers = current_centers.copy()
            
    return best_centers, best_r

def run_packing():
    """
    Main function to return the packing solution.
    """
    # Run optimization
    centers, r_equal = optimize_packing()
    
    # In the equal radius assumption, all radii are r_equal.
    # However, to maximize sum of radii, we can potentially have different radii.
    # But equal radii is a very strong lower bound and likely near optimal for sum.
    # Let's return equal radii for simplicity and robustness.
    # If we wanted to be fancy, we could compute Voronoi radii, but that's complex.
    
    radii = np.full(26, r_equal)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
