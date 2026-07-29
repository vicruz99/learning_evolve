# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 07a91dcd) state=f064cedf sum of radii=1.710593 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_radii(centers):
    """
    Computes the maximum possible radius for each circle given the centers.
    r_i is limited by the distance to the square boundaries and half the distance to the nearest other circle center.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Distance to boundaries
    x = centers[:, 0]
    y = centers[:, 1]
    dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Distance to other circles
    # Compute pairwise distance matrix
    # dist_matrix[i, j] = ||c_i - c_j||
    # We can compute this efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # For each circle i, find min distance to j != i
    # np.fill_diagonal(dist_matrix, np.inf) to ignore self-distance
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist_other = np.min(dist_matrix, axis=1)
    
    # Radius is limited by half the distance to the nearest neighbor
    radii = np.minimum(dist_boundary, 0.5 * min_dist_other)
    
    return radii

def objective_function(centers_flat):
    """
    Objective function to maximize sum of radii.
    Returns negative sum of radii for minimization.
    """
    centers = centers_flat.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)

def get_initial_centers(n_circles=26):
    """
    Generates initial centers in a hexagonal pattern.
    """
    # We want to fit 26 circles.
    # A 5x5 grid is 25 circles. A hexagonal packing can fit more.
    # Let's try to arrange them in rows.
    # 6 rows might work well.
    # Pattern: 4, 5, 4, 5, 4, 4 -> 26 circles? No, 4+5+4+5+4+4 = 26.
    # Or 5, 5, 5, 5, 4, 2?
    # Let's try a dense hexagonal packing.
    
    # Estimated radius for 26 circles in unit square if packed densely.
    # Area of square = 1. Density ~ 0.9. Total area ~ 0.9.
    # 26 * pi * r^2 = 0.9 => r ~ sqrt(0.9 / (26*pi)) ~ 0.105
    # Diameter ~ 0.21.
    # 1 / 0.21 ~ 4.7. So we can fit about 4-5 circles per row.
    
    rows = []
    # Try to fill rows with varying counts to approximate hexagonal packing
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 2 circles (to make 26) -> 5+5+5+5+4+2 = 26.
    # This seems unbalanced.
    
    # Let's try 6 rows of roughly 4-5 circles.
    # 4, 5, 4, 5, 4, 4 = 26.
    row_counts = [4, 5, 4, 5, 4, 4]
    
    centers = []
    
    # Spacing parameters. 
    # If we have 5 circles in a row, width is roughly 1. 
    # 5 circles diameter 2r. 5*2r <= 1 => r <= 0.1.
    # If r=0.1, horizontal spacing 0.2.
    # Vertical spacing for hex packing: r*sqrt(3) = 0.1732.
    # 6 rows height: 2r + 5*r*sqrt(3) = 0.2 + 0.866 = 1.066 > 1.
    # So r must be smaller if we use strict hex spacing for 6 rows.
    # But optimization will adjust.
    
    r_est = 0.1
    h_spacing = 2 * r_est
    v_spacing = r_est * np.sqrt(3)
    
    # Center the packing in the square
    # Calculate total width and height required for this pattern
    max_width = max(row_counts) * h_spacing
    total_height = 2 * r_est + (len(row_counts) - 1) * v_spacing
    
    # Adjust spacing to fit in 1x1 with some margin
    # Actually, let's just place them and let optimizer fix it.
    # Start with a reasonable scale.
    scale = 0.9 / (max(max_width, total_height)) # Fit roughly in 0.9 size
    h_spacing *= scale
    v_spacing *= scale
    r_est *= scale
    
    current_y = r_est # Start from bottom boundary + radius
    
    for r_idx, count in enumerate(row_counts):
        # Shift for hexagonal packing
        if r_idx % 2 == 1:
            x_offset = h_spacing / 2
        else:
            x_offset = 0
        
        # Calculate row width to center it
        row_width = (count - 1) * h_spacing
        start_x = (1 - row_width) / 2 - x_offset
        
        for c in range(count):
            x = start_x + c * h_spacing
            y = current_y
            centers.append([x, y])
        
        current_y += v_spacing
        
    centers = np.array(centers)
    # Ensure we have exactly 26
    if len(centers) < n_circles:
        # Add more if needed (should not happen with correct row_counts)
        while len(centers) < n_circles:
            centers = np.vstack([centers, [0.5, 0.5]])
    elif len(centers) > n_circles:
        centers = centers[:n_circles]
        
    return centers

def run_packing() -> tuple:
    n_circles = 26
    
    # Try multiple initial configurations to avoid local minima
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate initial hexagonal centers
    init_centers = get_initial_centers(n_circles)
    
    # Create a few variations by random perturbation
    variations = []
    variations.append(init_centers)
    
    np.random.seed(42)
    for _ in range(3):
        perturbed = init_centers + np.random.normal(0, 0.01, size=init_centers.shape)
        # Clip to valid range (with some padding)
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.05, 0.95)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.05, 0.95)
        variations.append(perturbed)
        
    # Also try a grid initialization
    grid_centers = np.zeros((n_circles, 2))
    # 5x5 grid is 25, add one
    # Let's try 6x5 grid subset?
    # Just a random uniform distribution might be good too?
    # Let's stick to structured ones.
    
    for i, centers in enumerate(variations):
        centers_flat = centers.flatten()
        
        # Bounds for coordinates [0, 1]
        bounds = [(0, 1) for _ in range(n_circles * 2)]
        
        # Use Powell method for derivative-free optimization
        # Maximize sum of radii => Minimize negative sum
        res = minimize(objective_function, centers_flat, method='Powell', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-8, 'xtol': 1e-8})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = res.x.reshape(-1, 2)
            
    # Compute radii for the best centers
    best_radii = compute_radii(best_centers)
    
    # Final validation and cleanup (ensure no negative radii or overlaps due to precision)
    # The compute_radii function guarantees validity based on centers, 
    # but we should clamp radii if any numerical issues occurred (unlikely).
    
    # Just to be safe, re-compute radii strictly
    best_radii = np.maximum(best_radii, 0.0)
    
    # Check for overlaps explicitly and fix if any (due to precision)
    # Though the radius definition prevents overlap, floating point errors might occur.
    # The validation function allows 1e-12 tolerance.
    
    return best_centers, best_radii, np.sum(best_radii)

# Function required by the prompt structure
def run_packing_solver():
    return run_packing()

# The prompt asks for run_packing function
# But the variable name in the prompt description is run_packing
# However, the validation checks the return of run_packing? 
# "You must define the run_packing function: def run_packing() -> tuple..."
# So I will define it as run_packing.
