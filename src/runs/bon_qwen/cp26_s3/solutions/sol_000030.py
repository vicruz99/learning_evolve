# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfc1b343) state=fa75519d sum of radii=0.000081 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

# Constants
N_CIRCLES = 26
TOLERANCE = 1e-9
BINARY_SEARCH_ITERATIONS = 25
OPTIMIZATION_ITERATIONS = 2000

def get_hexagonal_initial_centers(n):
    """
    Generates an initial configuration of centers based on a hexagonal lattice.
    This provides a structured, dense starting point.
    """
    centers = []
    # Parameters for layout: trying to fit circles with approximate diameter 0.2
    y = 0.1
    row = 0
    
    while len(centers) < n and y < 0.95:
        # Alternate row starts for hexagonal packing
        if row % 2 == 0:
            x_start = 0.1
            count = 5 # Fit 5 circles in width 1 (0.1 to 0.9 step 0.2)
        else:
            x_start = 0.2
            count = 4 # Fit 4 circles in shifted row
            
        for k in range(count):
            if len(centers) < n:
                x = x_start + k * 0.2
                if x <= 0.95: 
                     centers.append([x, y])
        
        y += 0.1732 # Vertical spacing for hexagonal packing (sqrt(3)/2 * 0.2)
        row += 1
        
    # Fill remaining with random points if we didn't get enough
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
        
    return np.array(centers[:n])

def get_grid_initial_centers(n):
    """
    Generates an initial configuration based on a square grid.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    padding = 0.05
    width = 1 - 2 * padding
    height = 1 - 2 * padding
    
    centers = []
    for r in range(rows):
        for c in range(cols):
            if len(centers) < n:
                if cols > 1:
                    x = padding + c * (width / (cols - 1))
                else:
                    x = 0.5
                if rows > 1:
                    y = padding + r * (height / (rows - 1))
                else:
                    y = 0.5
                centers.append([x, y])
    
    return np.array(centers)

def get_random_initial_centers(n):
    """
    Generates random initial centers within a safe margin.
    """
    return np.random.uniform(0.1, 0.9, (n, 2))

def calculate_violation(centers, r):
    """
    Calculates the total violation of constraints (overlaps and boundary breaches).
    """
    n = centers.shape[0]
    violation = 0.0
    
    # Circle-Circle overlaps
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Overlap occurs if distance < 2r
    overlaps = 2 * r - dists
    overlaps = np.maximum(0, overlaps)
    violation += np.sum(overlaps**2)
    
    # Boundary violations
    # x < r  => r - x > 0
    violation += np.sum(np.maximum(0, r - centers[:, 0])**2)
    # x > 1 - r => x - (1-r) > 0
    violation += np.sum(np.maximum(0, centers[:, 0] - (1 - r))**2)
    # y < r
    violation += np.sum(np.maximum(0, r - centers[:, 1])**2)
    # y > 1 - r
    violation += np.sum(np.maximum(0, centers[:, 1] - (1 - r))**2)
    
    return violation

def objective_function(centers_flat, r):
    """
    Objective function wrapper for scipy.optimize.
    """
    centers = centers_flat.reshape(-1, 2)
    return calculate_violation(centers, r)

def try_pack(r, init_centers_list):
    """
    Attempts to find a valid packing for a given radius r using multiple initial seeds.
    Returns (success, best_centers).
    """
    bounds = [(0.0, 1.0)] * (N_CIRCLES * 2)
    best_violation = float('inf')
    best_centers = None
    
    for init_centers in init_centers_list:
        x0 = init_centers.flatten()
        
        try:
            res = opt.minimize(
                objective_function, 
                x0, 
                args=(r,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': OPTIMIZATION_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_violation:
                best_violation = res.fun
                best_centers = res.x.reshape(-1, 2)
                
                # If we found a nearly perfect packing, return immediately
                if res.fun < TOLERANCE:
                    return True, best_centers
                    
        except Exception:
            continue
            
    # Check if the best found solution is within a reasonable tolerance
    if best_violation < 1e-6:
        return True, best_centers
    
    return False, best_centers

def run_packing():
    """
    Main function to solve the circle packing problem by maximizing the sum of radii.
    Uses binary search on the radius combined with local optimization.
    """
    # 1. Prepare initial configurations for diversity
    inits = []
    inits.append(get_hexagonal_initial_centers(N_CIRCLES))
    inits.append(get_grid_initial_centers(N_CIRCLES))
    for _ in range(5):
        inits.append(get_random_initial_centers(N_CIRCLES))
        
    low = 0.08
    high = 0.12 # Upper bound estimate
    best_r = low
    best_centers = inits[0]
    
    # Establish a baseline valid configuration with a small radius
    # This ensures we have a starting point for the continuation method
    test_r = 0.02
    success, temp_centers = try_pack(test_r, inits)
    if success:
        best_centers = temp_centers
        best_r = test_r
    else:
        # Fallback if initial small pack fails (unlikely)
        pass

    # 2. Binary Search for the optimal radius
    for i in range(BINARY_SEARCH_ITERATIONS):
        mid = (low + high) / 2.0
        
        # Include the best configuration found so far as a seed (Continuation Method)
        current_inits = list(inits)
        if best_centers is not None:
            current_inits.append(best_centers)
            
        success, centers = try_pack(mid, current_inits)
        
        if success:
            best_r = mid
            best_centers = centers
            low = mid + 1e-5 # Try larger radius
        else:
            high = mid - 1e-5 # Radius too large, try smaller
            
        if high - low < 1e-5:
            break
            
    # 3. Final Validation and Adjustment
    centers = best_centers
    r_final = best_r
    
    # Calculate violation for the best found radius
    violation = calculate_violation(centers, r_final)
    
    # If violation is significant, shrink radius to ensure strict validity
    if violation > 1e-8:
        # Find the largest r <= r_final that is valid for this specific center configuration
        r_low = 0.0
        r_high = r_final
        for _ in range(50):
            r_mid = (r_low + r_high) / 2
            if calculate_violation(centers, r_mid) < 1e-9:
                r_low = r_mid
            else:
                r_high = r_mid
        r_final = r_low

    radii = np.full(N_CIRCLES, r_final)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
