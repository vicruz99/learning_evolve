import numpy as np
from scipy.optimize import differential_evolution

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We attempt to place circles in a hexagonal pattern.
    # A hexagonal packing has a density of pi/sqrt(12).
    # We estimate the number of rows. sqrt(26) ~ 5.1. 
    # Let's try 6 rows. 
    # Pattern: 5, 4, 5, 4, 5, 4 ... sums to 27? 
    # 5+5+5+5+5+1 = 26?
    # Let's try a grid first for a baseline, then optimize.
    # Actually, a random or grid start is fine for DE, but a lattice is better.
    
    centers = np.zeros((n_circles, 2))
    
    # Let's try a 5x5 grid (25 circles) plus 1 extra, then optimize.
    # Or a hexagonal arrangement.
    # Hexagonal rows: 
    # Row 0: 5 circles
    # Row 1: 4 circles
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles? Total 26.
    
    # Let's construct a hexagonal grid
    row_counts = [5, 4, 5, 4, 5, 3] # Sum = 26
    # Check width: 5 circles -> width ~ 10r. If r~0.1, width 1.
    # Check height: 6 rows. Spacing sqrt(3)/2 * 2r = sqrt(3)r ~ 1.732r.
    # 5 gaps -> 5 * 1.732r ~ 8.66r. Plus 2r margins = 10.66r. 
    # If r=0.1, height 1.066 > 1. So r must be slightly less than 0.1.
    # This is a good starting point.
    
    idx = 0
    # Estimated radius for initialization
    r_est = 0.095 
    spacing_x = 2 * r_est
    spacing_y = np.sqrt(3) * r_est
    
    for r_idx, count in enumerate(row_counts):
        y = r_est + r_idx * spacing_y
        
        # X positions
        if r_idx % 2 == 0:
            # Start at r_est
            start_x = r_est
        else:
            # Shifted by r_est (half spacing_x)
            start_x = r_est + r_est # = 2*r_est
            
        for c_idx in range(count):
            x = start_x + c_idx * spacing_x
            centers[idx] = [x, y]
            idx += 1
            
    # Ensure we have exactly 26 centers
    centers = centers[:n_circles]

    def objective(centers_flat):
        """
        Objective function to minimize (negative sum of radii).
        Centers are flattened array of size 2*n_circles.
        """
        centers_2d = centers_flat.reshape((n_circles, 2))
        
        # Clip centers to [0, 1] to avoid invalid states, though optimization bounds should handle it.
        # But for distance calc, we assume centers are valid.
        
        radii = np.zeros(n_circles)
        
        for i in range(n_circles):
            x, y = centers_2d[i]
            
            # Distance to boundaries
            dist_bound = min(x, 1 - x, y, 1 - y)
            if dist_bound < 0: 
                # If center is outside, radius is negative? 
                # We should penalize heavily or clip.
                # But DE bounds keep it in [0,1].
                dist_bound = 0 
            
            min_dist_to_other = 2.0 # Infinity
            
            for j in range(n_circles):
                if i == j:
                    continue
                dx = centers_2d[i, 0] - centers_2d[j, 0]
                dy = centers_2d[i, 1] - centers_2d[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist_to_other:
                    min_dist_to_other = dist
            
            # Radius is limited by distance to boundary and half distance to nearest neighbor
            # Note: r_i <= dist_bound AND r_i + r_j <= dist_ij
            # If we set r_i = min(dist_bound, 0.5 * min_dist_to_other), 
            # then r_i + r_j <= 0.5*d_ij + 0.5*d_ij = d_ij. So no overlap.
            
            r_i = min(dist_bound, 0.5 * min_dist_to_other)
            radii[i] = r_i
            
        return -np.sum(radii)

    # Optimization bounds: each coordinate in [0, 1]
    bounds = [(0, 1)] * (2 * n_circles)
    
    # Use Differential Evolution
    # maxiter can be increased if needed, but time is limited.
    # seed for reproducibility
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=15, 
            mutation=(0.5, 1.5), 
            recombination=0.7,
            seed=42,
            init='latinhypercube'
        )
        
        best_centers = result.x.reshape((n_circles, 2))
        
        # Recalculate radii based on optimal centers
        final_radii = np.zeros(n_circles)
        for i in range(n_circles):
            x, y = best_centers[i]
            dist_bound = min(x, 1 - x, y, 1 - y)
            min_dist_to_other = 2.0
            for j in range(n_circles):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < min_dist_to_other:
                    min_dist_to_other = dist
            final_radii[i] = min(dist_bound, 0.5 * min_dist_to_other)
            
        sum_radii = np.sum(final_radii)
        
        return best_centers, final_radii, sum_radii
        
    except Exception as e:
        # Fallback to initial guess if optimization fails
        final_radii = np.zeros(n_circles)
        for i in range(n_circles):
            x, y = centers[i]
            dist_bound = min(x, 1 - x, y, 1 - y)
            min_dist_to_other = 2.0
            for j in range(n_circles):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < min_dist_to_other:
                    min_dist_to_other = dist
            final_radii[i] = min(dist_bound, 0.5 * min_dist_to_other)
            
        sum_radii = np.sum(final_radii)
        return centers, final_radii, sum_radii