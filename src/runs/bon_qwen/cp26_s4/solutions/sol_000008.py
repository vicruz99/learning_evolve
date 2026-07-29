# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ca1ebfe6) state=cd68d1ee sum of radii=1.768167 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Function to calculate the maximum feasible radius for a given set of centers
    # r is limited by the distance to boundaries and distance to other centers
    def max_radius(centers):
        # Distance to boundaries
        # x - r >= 0 => r <= x
        # 1 - x - r >= 0 => r <= 1 - x
        # same for y
        r_boundary = np.minimum(
            np.minimum(centers[:, 0], 1 - centers[:, 0]),
            np.minimum(centers[:, 1], 1 - centers[:, 1])
        )
        min_r_boundary = np.min(r_boundary)
        
        # Distance to other centers
        # dist >= 2r => r <= dist / 2
        # Compute pairwise distances
        # To avoid O(N^2) explosion in memory, compute iteratively or vectorized
        # N=26 is small, so vectorized is fine
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf) # Ignore self-distance
        min_dist = np.min(dists)
        min_r_centers = min_dist / 2.0
        
        return min(min_r_boundary, min_r_centers)

    # Objective function for optimization: Negative of max radius
    # We want to maximize r, so minimize -r
    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        r = max_radius(centers)
        return -r # We minimize negative radius

    # Helper to generate hexagonal lattice initial positions
    def generate_hex_lattice():
        # Try to fit rows. 6+5+6+5+4 = 26
        # Rows at y = r, r + sqrt(3)r, ...
        # But we don't know r yet. Let's assume a loose r=0.1
        # and scale later.
        r_init = 0.09
        points = []
        
        row_counts = [6, 5, 6, 5, 4]
        y = r_init
        for count in row_counts:
            # x coordinates for this row
            # Total width for 'count' circles is count * 2*r
            # We want to center them or align left? 
            # Hexagonal packing usually shifts odd/even rows.
            # Let's align centers such that they touch.
            # First circle at x = r_init (touching left wall)
            # Then x += 2*r_init
            
            # To center the row in [0, 1], calculate total width
            width = count * 2 * r_init
            start_x = (1 - width) / 2 + r_init # Shift so first circle touches left? 
            # Actually, for hex packing, rows are offset.
            # Let's just place them touching each other horizontally.
            # And center the whole row horizontally.
            
            # x coordinates relative to start
            xs = np.linspace(start_x, start_x + (count-1)*2*r_init, count)
            
            # If row index is odd (1, 3...), shift x by r_init?
            # In standard hex packing, row k+1 is shifted by r relative to row k.
            if len(points) % 2 == 1:
                xs += r_init
            
            for x in xs:
                points.append([x, y])
            
            y += math.sqrt(3) * r_init
            
        return np.array(points)

    best_centers = None
    best_r = 0
    
    # Strategy 1: Optimize from hex lattice
    centers_init = generate_hex_lattice()
    
    # Bounds for centers: [0, 1]
    bounds = [(0, 1)] * (2 * n)
    
    # Run optimization
    # Nelder-Mead is good for non-smooth functions (min function is non-smooth)
    result = minimize(objective, centers_init.flatten(), method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 10000})
    
    centers_opt = result.x.reshape(-1, 2)
    r_opt = -result.fun
    best_centers = centers_opt
    best_r = r_opt
    
    # Strategy 2: Random restarts to avoid local minima
    # Try a few random configurations
    for _ in range(10):
        # Random centers
        rng = np.random.default_rng(42 + _)
        centers_rand = rng.uniform(0.1, 0.9, size=(n, 2))
        
        result_rand = minimize(objective, centers_rand.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000})
        
        if -result_rand.fun > best_r:
            best_r = -result_rand.fun
            best_centers = result_rand.x.reshape(-1, 2)

    # Strategy 3: Differential Evolution for global search (slower but robust)
    # Since N is small (52 dims), this might take time but let's try with limited iterations
    try:
        # DE is slow, limit population size and generations
        res_de = differential_evolution(objective, bounds, seed=42, 
                                        maxiter=200, popsize=15, 
                                        tol=1e-5, polish=True)
        if -res_de.fun > best_r:
            best_r = -res_de.fun
            best_centers = res_de.x.reshape(-1, 2)
    except Exception:
        pass

    # Final validation and radii assignment
    # The optimization maximizes the minimum feasible radius r_min.
    # We set all radii to this r_min.
    final_r = max_radius(best_centers)
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii
