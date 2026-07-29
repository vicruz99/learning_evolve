# sol_000253 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 02c202ea) state=81d908e5 sum of radii=2.210216 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def compute_sum_radii(coords):
    """
    Computes the sum of radii for a given set of centers.
    Radii are determined by the closest constraint (boundary or other circle).
    """
    centers = np.array(coords).reshape(26, 2)
    n = 26
    radii = np.zeros(n)
    
    # Calculate pairwise distances matrix to speed up
    # dist_matrix[i, j] = distance between circle i and j
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    for i in range(n):
        x, y = centers[i]
        # Distance to boundaries
        # Center must be at least r away from 0 and 1
        # So r <= x, r <= 1-x, r <= y, r <= 1-y
        dist_to_bound = min(x, 1.0 - x, y, 1.0 - y)
        
        # Distance to other circles
        # r_i + r_j <= dist_ij => r_i <= dist_ij - r_j
        # But here we assume we are calculating the max possible radius for circle i
        # given fixed centers. However, radii are coupled.
        # But if we just want to evaluate the sum of max radii for a FIXED configuration of centers,
        # the radius of circle i is limited by dist_ij / 2 assuming other circles are small?
        # No, that's not right.
        # Actually, if centers are fixed, the condition "non-overlapping" means
        # r_i + r_j <= dist_ij.
        # We want to maximize sum r_i subject to r_i + r_j <= dist_ij and r_i <= dist_bound_i.
        # This is a linear programming problem for fixed centers.
        # But usually, for dense packings, r_i is approx dist_ij / 2.
        # Let's use the standard approximation: r_i = min(dist_bound, min_j(dist_ij / 2))
        # This is a valid packing (might not be the max sum for fixed centers, but it's a valid point in search space)
        # And for equal circles, it's optimal.
        
        min_dist_to_other = np.min(dist_matrix[i, :]) # Includes dist to self (0), so ignore
        # Actually dist_matrix[i, i] is 0. We need min over j != i
        min_dist_to_other = np.min([d for j, d in enumerate(dist_matrix[i, :]) if j != i])
        
        r_i = min(dist_to_bound, min_dist_to_other / 2.0)
        radii[i] = r_i
        
    return np.sum(radii)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Generate initial hexagonal packing
    # Estimate radius for 26 circles in unit square. 
    # 5x5 grid r=0.1. Hexagonal allows slightly better density.
    # Let's try to fit a hexagonal grid.
    # Rows offset.
    
    # Heuristic for initial radius
    r_est = 0.1
    
    # Generate points
    points = []
    row = 0
    col = 0
    x_spacing = 2 * r_est
    y_spacing = r_est * np.sqrt(3)
    
    # Try to fill the square
    # We can iterate over a grid and pick points that fit or just generate a cloud
    # A better way: Generate a large hexagonal grid, pick 26 points that are well distributed
    
    # Let's create a grid of potential centers
    # Range x: [r_est, 1-r_est], y: [r_est, 1-r_est]
    # But we can be looser initially.
    
    # Create a list of points
    candidates = []
    y = 0.1
    while y <= 0.9:
        x = 0.1
        offset = 0.0 if int((y - 0.1) / y_spacing) % 2 == 0 else r_est # Offset rows
        # Actually standard hex offset is half x_spacing
        offset = 0.0
        if int((y - 0.1) / y_spacing) % 2 == 1:
            offset = r_est # half of 2*r_est
            
        while x <= 0.9:
            candidates.append([x + offset, y])
            x += x_spacing
        y += y_spacing
    
    # If we don't have enough candidates, add more or reduce spacing
    if len(candidates) < n:
        # Reduce r_est and retry
        r_est = 0.08
        candidates = []
        y = 0.1
        while y <= 0.9:
            x = 0.1
            offset = 0.0
            if int((y - 0.1) / (r_est * np.sqrt(3))) % 2 == 1:
                offset = r_est
            
            while x <= 0.9:
                candidates.append([x + offset, y])
                x += 2 * r_est
            y += r_est * np.sqrt(3)
    
    # Select n points
    # We can just take the first n, or distribute them
    # To ensure good coverage, let's pick points with some randomness or just the first n
    # If candidates > n, we should pick a subset that maximizes min distance?
    # For now, just take first n and optimize.
    # Better: Pick n points uniformly from candidates if len > n?
    # Or just use a random initialization near a grid.
    
    # Let's just use a random initialization with repulsion to be safe, 
    # or use the grid points.
    # Grid points are usually good starts.
    
    if len(candidates) >= n:
        # Pick n points. To avoid clustering, maybe pick from different parts?
        # Just taking first n might be clustered in one corner if grid generation was weird.
        # But my generation fills the square.
        # Let's shuffle and pick n
        np.random.shuffle(candidates)
        initial_centers = np.array(candidates[:n])
    else:
        # Fallback random
        initial_centers = np.random.rand(n, 2) * 0.8 + 0.1 # Keep away from edges initially

    # 2. Optimize
    # We want to maximize sum_radii. Minimize negative sum.
    # Use Nelder-Mead or Powell.
    # Since function is non-smooth, maybe basin hopping or multiple restarts?
    # But we have limited time/resources.
    # Let's try one run of Nelder-Mead from the grid start.
    
    # Flatten centers for scipy
    x0 = initial_centers.flatten()
    
    # Bounds: centers must be in [0, 1]
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Objective
    def objective(x):
        return -compute_sum_radii(x)
    
    # Optimization
    # Nelder-Mead is good for non-smooth
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_centers = result.x.reshape(n, 2)
    best_sum = -result.fun
    
    # Calculate radii for the best centers
    # We need to compute radii correctly for the final output
    # The compute_sum_radii function used a heuristic (min dist/2).
    # For the final output, we should ensure valid packing.
    # Actually, the heuristic r_i = min(bound, min_dist/2) produces a valid packing
    # where circles might be smaller than necessary if neighbors are small.
    # But since we maximized the sum of these radii, this is a valid solution.
    # However, we might be able to increase radii further if we solve the LP for radii?
    # No, the objective was exactly that sum.
    
    # Let's re-calculate radii carefully to ensure validity and report sum.
    centers = best_centers
    radii = np.zeros(n)
    
    # Distance matrix
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
    for i in range(n):
        x, y = centers[i]
        d_bound = min(x, 1-x, y, 1-y)
        d_min = np.inf
        for j in range(n):
            if i != j:
                if dists[i, j] < d_min:
                    d_min = dists[i, j]
        
        radii[i] = min(d_bound, d_min / 2.0)
        
    # Validate
    is_valid = validate_packing(centers, radii)
    if not is_valid:
        # Fallback to a known good simple packing if optimization failed (unlikely)
        # Or just return what we have, but ensure it's valid.
        # The logic above guarantees validity if centers are in [0,1] and radii calculated as min.
        # Wait, if radii[i] = dist_min/2, then radii[i] + radii[j] = dist/2 + dist/2 = dist.
        # So they touch. Valid.
        pass

    return centers, radii, np.sum(radii)

# Run to check
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(validate_packing(c, r))
