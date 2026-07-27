import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Greedy Maximin Distance
    # Create a grid of candidate points
    grid_size = 100
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    candidates = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    
    centers = []
    # Start with a random point or center to break symmetry
    # Using center (0.5, 0.5) is safe
    current_centers = [np.array([0.5, 0.5])]
    
    # Select remaining n-1 points
    for _ in range(n - 1):
        best_point = None
        max_min_dist = -1.0
        
        # Compute distances from each candidate to all existing centers
        # Vectorized distance calculation
        # candidates shape: (N, 2), current_centers shape: (k, 2)
        # dists shape: (N, k)
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - np.array(current_centers)[np.newaxis, :, :], axis=2)
        
        # Find min distance for each candidate
        min_dists = np.min(dists, axis=1)
        
        # Find index of candidate with largest min distance
        best_idx = np.argmax(min_dists)
        best_min_dist = min_dists[best_idx]
        
        if best_min_dist > max_min_dist:
            max_min_dist = best_min_dist
            best_point = candidates[best_idx]
            
        if best_point is not None:
            current_centers.append(best_point)
            
    centers = np.array(current_centers)
    
    # Initialize radii small enough to be valid
    # Max possible radius for a circle at (x,y) is min(x, 1-x, y, 1-y)
    # And half distance to nearest neighbor.
    # We'll set a safe small radius.
    radii = np.full(n, 0.01)
    
    # 2. Optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Size 3 * n = 78
    initial_vars = np.zeros(3 * n)
    for i in range(n):
        initial_vars[3 * i] = centers[i, 0]
        initial_vars[3 * i + 1] = centers[i, 1]
        initial_vars[3 * i + 2] = radii[i]
    
    # Bounds:
    # x, y in [0, 1]
    # r >= 0
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r (max radius is 0.5)
        
    # Penalty parameter
    penalty_weight = 1000.0
    
    def objective(vars):
        loss = 0.0
        penalty = 0.0
        
        # Extract variables
        # To avoid loop overhead in python, we can do vectorized ops if needed,
        # but n=26 is small enough for loops.
        
        # Accumulate negative sum of radii (we want to maximize sum)
        # And accumulate penalties
        
        # Boundary penalties
        for i in range(n):
            x = vars[3 * i]
            y = vars[3 * i + 1]
            r = vars[3 * i + 2]
            
            loss -= r # Maximize sum of radii
            
            # Boundary constraints: x-r >= 0 => r-x <= 0
            # violation = max(0, r - x)
            v1 = max(0.0, r - x)
            v2 = max(0.0, r - (1 - x))
            v3 = max(0.0, r - y)
            v4 = max(0.0, r - (1 - y))
            
            penalty += (v1**2 + v2**2 + v3**2 + v4**2)
            
        # Overlap penalties
        # Only check i < j
        for i in range(n):
            xi = vars[3 * i]
            yi = vars[3 * i + 1]
            ri = vars[3 * i + 2]
            
            for j in range(i + 1, n):
                xj = vars[3 * j]
                yj = vars[3 * j + 1]
                rj = vars[3 * j + 2]
                
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                overlap = ri + rj - dist
                if overlap > 0:
                    penalty += overlap**2
                    
        return loss + penalty_weight * penalty

    # Use L-BFGS-B
    result = minimize(objective, initial_vars, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract solution
    final_vars = result.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_vars[3 * i]
        final_centers[i, 1] = final_vars[3 * i + 1]
        final_radii[i] = final_vars[3 * i + 2]
        
    # Clamp radii to non-negative just in case
    final_radii = np.maximum(final_radii, 0.0)
    
    # Validate
    if not validate_packing(final_centers, final_radii):
        # If invalid, fallback to a safe valid configuration
        # Though with high penalty it should be valid.
        # Fallback: simple grid
        fallback_centers = []
        fallback_radii = []
        count = 0
        step = 0.2
        for i in range(6):
            for j in range(5):
                if count < n:
                    x = 0.1 + i * step
                    y = 0.1 + j * step
                    r = 0.05
                    fallback_centers.append([x, y])
                    fallback_radii.append(r)
                    count += 1
        return np.array(fallback_centers), np.array(fallback_radii), sum(fallback_radii)

    return final_centers, final_radii, np.sum(final_radii)