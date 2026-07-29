# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4e4d202b) state=3d45e911 sum of radii=2.150206 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_max_radius(center, other_centers, other_radii):
    """
    Calculates the maximum possible radius for a circle at 'center'
    given other fixed circles.
    """
    x, y = center
    
    # Distance to boundaries
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    r_boundary = min(x, 1 - x, y, 1 - y)
    
    # Distance to other circles
    # r_i + r_j <= dist => r_i <= dist - r_j
    # We need r_i >= 0, so dist must be >= r_j (center cannot be inside other circle)
    # However, if we are inside, dist - r_j is negative.
    # The max radius would be negative, which is invalid, but mathematically
    # the function returns the limit. We want to maximize this.
    
    # Vectorized distance calculation
    diffs = other_centers - center
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Constraints from neighbors
    r_neighbors = dists - other_radii
    
    # The radius is limited by the tightest constraint
    return min(r_boundary, np.min(r_neighbors))

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    np.random.seed(42) # For reproducibility
    
    # --- Initialization ---
    # Start with random positions and very small radii
    centers = np.random.rand(n, 2)
    radii = np.full(n, 0.001) # Small enough to avoid initial overlaps
    
    # Ensure initial validity (just in case random points are super close)
    # With 0.001 radius, overlaps are unlikely but let's just be safe by not enforcing strictly here
    # The optimizer will handle it, but getting stuck in invalid state is bad.
    # 0.001 is small enough for 26 circles (avg dist ~ 0.2)
    
    # --- Optimization Loop ---
    # We perform multiple passes over all circles
    num_passes = 50 
    
    for _pass in range(num_passes):
        # Randomize order of circles to optimize to avoid bias
        indices = np.random.permutation(n)
        
        for idx in indices:
            # Separate current circle from others
            current_center = centers[idx].copy()
            current_radius = radii[idx]
            
            other_centers = np.delete(centers, idx, axis=0)
            other_radii = np.delete(radii, idx)
            
            # Define the objective function to maximize: max possible radius at a given center
            # We minimize the negative of this function
            def objective(pos):
                # pos is (x, y)
                # Calculate max radius allowed at this position
                # Note: This function is non-smooth due to min()
                r = get_max_radius(pos, other_centers, other_radii)
                return -r # Minimize negative radius
            
            # Initial guess is the current center
            x0 = current_center
            
            # Use Nelder-Mead simplex method (derivative free)
            # Bounds are [0, 1] for x and y
            bounds = [(0.0, 1.0), (0.0, 1.0)]
            
            try:
                result = opt.minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    tol=1e-7,
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                
                new_center = result.x
                new_radius = -result.fun
                
                # Check if we found a valid and improved configuration
                # The optimizer might find a point inside another circle (radius < 0)
                # but usually it will escape if started from a valid point.
                # We enforce radius >= 0.
                if new_radius < 1e-9:
                    new_radius = 0.0
                    # If radius becomes 0, maybe keep old? 
                    # But if we are stuck, 0 is the limit.
                
                # Update
                centers[idx] = new_center
                radii[idx] = new_radius
                
            except Exception as e:
                # If optimization fails, keep current state
                pass

    # Final validation and cleanup
    # Ensure no negative radii
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper for validation (provided in prompt, but good to have locally for testing if needed)
# Not included in run_packing to keep it clean, but the logic follows the rules.
