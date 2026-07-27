import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # Define objective function: minimize negative sum of radii
    def objective(vars):
        # vars layout: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # Radii are at indices 2, 5, 8, ...
        radii = vars[2::3]
        return -np.sum(radii)

    # Define constraints
    # We return an array of values that must be >= 0
    def constraint_function(vars):
        constraints = []
        
        # 1. Boundary Constraints: x-r >= 0, 1-(x+r) >= 0, y-r >= 0, 1-(y+r) >= 0
        for i in range(n_circles):
            idx = i * 3
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            constraints.extend([
                x - r,
                1.0 - (x + r),
                y - r,
                1.0 - (y + r)
            ])
        
        # 2. Non-overlap Constraints: dist^2 - (r1 + r2)^2 >= 0
        # We iterate over all unique pairs
        # To optimize, we can compute centers and radii arrays
        centers = vars.reshape(-1, 3)[:, :2] # (n, 2)
        radii = vars.reshape(-1, 3)[:, 2]   # (n,)
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                sum_r = radii[i] + radii[j]
                constraints.append(dist_sq - sum_r**2)
                
        return np.array(constraints)

    cons = {'type': 'ineq', 'fun': constraint_function}
    
    # Bounds: x, y in [0, 1], r in [0, 0.5] (loose upper bound for r)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n_circles

    best_sum_radii = 0.0
    best_vars = None

    # Helper to run optimization from a specific start point
    def try_optimize(x0):
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
            return res
        except Exception:
            return None

    # Strategy: Try multiple initial configurations
    # 1. Grid layout with small radii
    # 2. Perturbed grid
    # 3. Random valid placement
    
    attempts = []
    
    # Configuration 1: Grid
    grid_points = []
    for r in range(5):
        for c in range(5):
            grid_points.append((0.1 + r * 0.2, 0.1 + c * 0.2))
    # Add 26th point in a gap
    grid_points.append((0.2, 0.2)) 
    
    x0_grid = np.zeros(n_circles * 3)
    for i, (x, y) in enumerate(grid_points):
        x0_grid[i*3] = x
        x0_grid[i*3+1] = y
        x0_grid[i*3+2] = 0.005 # Small initial radius
    attempts.append(x0_grid)
    
    # Configuration 2: Perturbed Grid
    x0_perturbed = np.copy(x0_grid)
    # Add noise to positions
    noise_x = np.random.uniform(-0.02, 0.02, n_circles)
    noise_y = np.random.uniform(-0.02, 0.02, n_circles)
    x0_perturbed[0::3] += noise_x
    x0_perturbed[1::3] += noise_y
    # Clamp to bounds
    x0_perturbed[0::3] = np.clip(x0_perturbed[0::3], 0.01, 0.99)
    x0_perturbed[1::3] = np.clip(x0_perturbed[1::3], 0.01, 0.99)
    attempts.append(x0_perturbed)

    # Configuration 3: Random valid placement (simple rejection sampling)
    # Just scatter points with small r
    np.random.seed(42) # For reproducibility
    random_centers = []
    for _ in range(n_circles):
        while True:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Check distance to existing points (approx check for validity with r=0.01)
            valid = True
            for cx, cy in random_centers:
                if (x-cx)**2 + (y-cy)**2 < (0.02)**2:
                    valid = False
                    break
            if valid:
                random_centers.append((x, y))
                break
    
    x0_random = np.zeros(n_circles * 3)
    for i, (x, y) in enumerate(random_centers):
        x0_random[i*3] = x
        x0_random[i*3+1] = y
        x0_random[i*3+2] = 0.005
    attempts.append(x0_random)

    # Run optimization for each attempt
    for x0 in attempts:
        res = try_optimize(x0)
        if res is not None and res.success:
            # Calculate sum of radii for this result
            current_radii = res.x[2::3]
            current_sum = np.sum(current_radii)
            
            # Verify validity manually just in case (optional but good practice)
            # The solver should satisfy constraints, but numerical issues can occur.
            # We check if the solution is "valid enough" by checking constraints.
            # However, the problem asks for a valid packing.
            # Let's trust the solver but check if constraints are satisfied within tolerance.
            
            # Re-evaluate constraints to be sure
            try:
                c_val = constraint_function(res.x)
                if np.all(c_val >= -1e-7): # Tolerance
                     if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_vars = res.x
            except:
                pass

    # If optimization didn't yield a valid result better than 0, fallback to simple grid
    if best_vars is None:
        # Fallback: Grid with radius 0.05 (just to return something valid)
        best_vars = np.zeros(n_circles * 3)
        idx = 0
        for r in range(5):
            for c in range(5):
                best_vars[idx] = 0.1 + r*0.2
                best_vars[idx+1] = 0.1 + c*0.2
                best_vars[idx+2] = 0.05
                idx += 3
        # 26th
        best_vars[idx] = 0.2
        best_vars[idx+1] = 0.2
        best_vars[idx+2] = 0.05
        best_sum_radii = 26 * 0.05

    # Extract results
    centers = best_vars.reshape(-1, 3)[:, :2]
    radii = best_vars.reshape(-1, 3)[:, 2]
    
    # Final validation and fixing
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    # Ensure centers are within bounds adjusted for radius
    # If a circle is slightly out, clip it and reduce radius if needed
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        
        # Clamp center
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        centers[i] = (x, y)
        
        # Adjust radius if it violates bounds
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            radii[i] = max_r

    # Fix overlaps by slightly reducing radii if necessary
    # This is a safety net.
    for _ in range(10): # Iterative fix
        max_overlap = 0.0
        overlap_pair = (-1, -1)
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    overlap = req_dist - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                        overlap_pair = (i, j)
        
        if max_overlap > 1e-7:
            i, j = overlap_pair
            # Reduce radii of overlapping circles proportionally
            factor = 0.5 # Just a heuristic reduction
            # Better: scale radii down so they touch
            # r1' + r2' = dist
            # Keep ratio? Or reduce both?
            # Let's just reduce the sum of radii involved to satisfy constraint
            current_sum_r = radii[i] + radii[j]
            new_sum_r = dist
            scale = new_sum_r / current_sum_r if current_sum_r > 0 else 0
            radii[i] *= scale
            radii[j] *= scale
            # Re-clip bounds
            for k in [i, j]:
                x, y = centers[k]
                r = radii[k]
                x = np.clip(x, r, 1.0 - r)
                y = np.clip(y, r, 1.0 - y) # typo fix below
                y = np.clip(y, r, 1.0 - r)
                centers[k] = (x, y)
        else:
            break
            
    final_sum = np.sum(radii)
    return centers, radii, final_sum