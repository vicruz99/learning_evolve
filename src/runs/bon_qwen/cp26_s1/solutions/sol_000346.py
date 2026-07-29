# sol_000346 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2c580e0d) state=031b03f4 sum of radii=2.026497 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Returns centers, radii, and the sum of radii.
    """
    n = 26
    
    # 1. Initialization Strategy
    # We aim for a hexagonal-like packing to achieve higher density than a square grid.
    # A square grid (5x5) gives r=0.1. We want r > 0.1.
    # A hexagonal pattern with 5 rows allows for tighter packing.
    # Distribution of 26 circles into 5 rows: 6, 5, 5, 6, 4 (Total 26)
    # Note: Standard hex packing shifts odd rows.
    
    row_counts = [6, 5, 5, 6, 4]
    
    # Initial guess for radius. We start slightly lower than the target to ensure validity.
    # Target r approx 0.10138. Let's start with 0.09.
    r_init = 0.09
    
    centers = []
    y = r_init
    
    for i, count in enumerate(row_counts):
        # Hexagonal offset: every other row is shifted horizontally
        if i % 2 == 1:
            x_start = 2 * r_init # Shifted row starts at 2r (radius gap + radius)
            # Actually, to center it nicely or fit max width, we calculate based on available space
            # Available width is 1 - 2*r. 
            # In a shifted row, the circles are nestled in the gaps.
            # A simpler initialization: place them evenly spaced in [r, 1-r]
            # But shifted rows in hex packing usually have centers at x = 2r, 4r...
            # Let's just distribute them evenly to start, the optimizer will fix it.
            x_coords = np.linspace(r_init, 1 - r_init, count)
        else:
            x_coords = np.linspace(r_init, 1 - r_init, count)
            
        for x in x_coords:
            centers.append([x, y])
        
        # Vertical spacing for hex packing is r * sqrt(3)
        y += r_init * math.sqrt(3)
    
    centers = np.array(centers)

    # 2. Optimization Function
    # We want to maximize the minimum distance between any two circles.
    # Let d_min be the minimum distance. The max radius r = d_min / 2.
    # To use scipy minimize (which minimizes), we minimize the negative of the minimum distance.
    # We also need to respect boundaries. 
    # We can penalize boundary violations in the objective or use constraints.
    # A robust way for packing is to minimize a "pressure" function or simply 
    # maximize min_dist while keeping centers inside [r, 1-r]. 
    # Since r varies, it's easier to fix a "target" r and check feasibility, 
    # but here we optimize positions for a dynamic r.
    
    # Alternative approach: Optimize positions to maximize the minimum separation.
    # Then calculate r based on that separation and boundaries.
    
    def objective(vars):
        """
        Negative of the minimum pairwise distance.
        vars is a 1D array of size 2*n (x1, y1, x2, y2, ...)
        """
        pos = vars.reshape(-1, 2)
        
        # Calculate all pairwise distances
        # Using broadcasting for efficiency
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        
        # Also consider boundary constraints. 
        # We want points to be as far from boundaries as possible?
        # No, we just need them inside [0,1]. 
        # If we just maximize min_dist between circles, we might push them to corners.
        # But we need to ensure they fit.
        # The actual achievable radius is min(min_dist/2, min_x_dist, min_y_dist_from_boundary)
        # But since we optimize positions, we can just maximize min_dist between circles
        # and let the final radius be determined by the tightest constraint.
        # However, to prevent circles from flying to corners and ignoring each other,
        # we can add a small penalty or just rely on the fact that dense packing maximizes sum.
        
        # Actually, simply maximizing min_dist between centers is sufficient 
        # if we assume the final radius is min(min_dist/2, distance to boundary).
        # But if circles go to corners, distance to boundary is small, limiting r.
        # The optimizer will naturally balance this if we penalize boundary proximity?
        # No, standard max-min packing usually works by just maximizing inter-circle distance.
        # Let's stick to maximizing min_dist between circles.
        
        return -min_dist

    # Initial position vector
    x0 = centers.flatten()

    # Bounds for coordinates: [0, 1]
    # Though strictly, they must be in [r, 1-r]. Since r is small, [0,1] is a safe initial bound.
    bounds = [(0, 1) for _ in range(2 * n)]

    # Run optimization
    # Nelder-Mead or BFGS? BFGS needs gradients. We can use 'Nelder-Mead' or 'Powell' for derivative-free.
    # Or provide a gradient? Numerical gradient is fine.
    # Let's use 'Nelder-Mead' for robustness, or 'BFGS' with finite difference.
    # Given the smoothness of distance, BFGS is often faster.
    
    res = minimize(objective, x0, method='BFGS', bounds=bounds, options={'maxiter': 2000})
    
    optimized_centers = res.x.reshape(-1, 2)

    # 3. Calculate Maximum Feasible Radius
    # The radius is limited by:
    # 1. Distance to other circles: r <= dist(i,j) / 2
    # 2. Distance to boundaries: r <= x_i, r <= 1-x_i, r <= y_i, r <= 1-y_i
    
    min_dist_to_boundary = np.inf
    min_dist_to_circle = np.inf
    
    # Distances to boundaries
    dists_x = np.minimum(optimized_centers[:, 0], 1 - optimized_centers[:, 0])
    dists_y = np.minimum(optimized_centers[:, 1], 1 - optimized_centers[:, 1])
    min_dist_to_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    
    # Distances between circles
    diff = optimized_centers[:, np.newaxis, :] - optimized_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_to_circle = np.min(dists) / 2.0
    
    # The maximum valid radius
    r = min(min_dist_to_boundary, min_dist_to_circle)
    
    # Add a tiny epsilon to avoid numerical errors in validation, 
    # but ensure strict inequality for the validation function logic (which uses -1e-12)
    # The validation checks: dist < r1+r2 - 1e-12.
    # If we set r exactly to min_dist/2, dist = 2r, so dist < 2r - 1e-12 is False.
    # It's safe. But to be safe from float errors, we can reduce r slightly.
    r = r * 0.999999
    
    radii = np.full(n, r)
    sum_radii = np.sum(radii)
    
    # Re-center if necessary? No, centers are optimized.
    # Just ensure centers are within bounds relative to r.
    # If optimization pushed a center to 0, and r > 0, it's invalid.
    # The optimization bounds were [0,1], but we didn't enforce [r, 1-r] dynamically.
    # However, if r is derived from min_dist_to_boundary, it ensures fit.
    # But we must check if centers are valid.
    
    # Sanity check: if any center is outside [r, 1-r], clamp or adjust?
    # Actually, if min_dist_to_boundary was the limiting factor, then centers are valid.
    # If min_dist_to_circle was limiting, then min_dist_to_boundary >= r, so centers are valid.
    
    return optimized_centers, radii, sum_radii

# Run the function to generate the solution
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radius: {r[0]}")
