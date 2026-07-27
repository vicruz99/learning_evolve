# sol_000208 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cccf4974) state=ebe5fbe7 sum of radii=1.789176 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def calculate_radii(centers):
        """
        Calculates the maximum valid radius for each circle given their centers.
        The radius is limited by the square boundaries and other circles.
        """
        radii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            # Distance to boundaries
            dist_bound = min(x, 1 - x, y, 1 - y)
            
            # Distance to nearest neighbor (half distance)
            min_dist = 1.0 # Initialize with a safe large value (max diameter is 1)
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    min_dist = d
            dist_neighbor = min_dist / 2.0
            
            # The valid radius is the minimum of boundary and neighbor constraints
            radii[i] = min(dist_bound, dist_neighbor)
        return radii

    def objective(centers_flat):
        """
        Objective function to minimize (negative sum of radii).
        """
        centers = centers_flat.reshape((n, 2))
        radii = calculate_radii(centers)
        return -np.sum(radii)

    def generate_hexagonal_init(r_init):
        """
        Generates a hexagonal packing initialization.
        """
        centers = np.zeros((n, 2))
        count = 0
        # Hexagonal parameters
        dx = 2 * r_init
        dy = np.sqrt(3) * r_init
        
        y = r_init
        row = 0
        while count < n:
            # Shift x for odd rows
            x_start = r_init if row % 2 == 0 else 2 * r_init
            x = x_start
            while x <= 1 - r_init + 1e-9:
                if count < n:
                    centers[count] = [x, y]
                    count += 1
                x += dx
            y += dy
            row += 1
        return centers

    best_sum_r = 0
    best_centers = None
    best_radii = None
    
    # Try multiple initializations to find a good local optimum
    # 1. Hexagonal packing with r=0.09 (fits well)
    # 2. Random valid configurations
    
    initial_configs = []
    
    # Config 1: Hexagonal
    centers_h = generate_hexagonal_init(0.09)
    initial_configs.append(centers_h)
    
    # Config 2: Random valid packing
    # Place circles with small radius to ensure no overlap, then let optimizer grow them
    rng = np.random.default_rng(42)
    for _ in range(5): # 5 random restarts
        temp_centers = np.zeros((n, 2))
        placed = 0
        attempts = 0
        r_temp = 0.04 # Small initial radius
        
        while placed < n and attempts < 10000:
            cx = rng.uniform(r_temp, 1 - r_temp)
            cy = rng.uniform(r_temp, 1 - r_temp)
            
            # Check overlap with already placed circles
            overlap = False
            for k in range(placed):
                dx = cx - temp_centers[k, 0]
                dy = cy - temp_centers[k, 1]
                if dx*dx + dy*dy < (2*r_temp)**2:
                    overlap = True
                    break
            
            if not overlap:
                temp_centers[placed] = [cx, cy]
                placed += 1
            attempts += 1
            
        if placed == n:
            initial_configs.append(temp_centers)

    # Optimization
    bounds = [(0, 1) for _ in range(2 * n)] # x, y for each circle
    
    for i, init_centers in enumerate(initial_configs):
        x0 = init_centers.flatten()
        
        # Use Powell method as it's derivative-free and robust for non-smooth objectives
        # Nelder-Mead is also good but Powell can sometimes be faster for high dimensions
        try:
            res = minimize(objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
            
            current_sum = -res.fun
            
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_centers = res.x.reshape((n, 2))
                # Recalculate radii to be precise
                best_radii = calculate_radii(best_centers)
        except Exception:
            continue
            
    # Fallback if optimization failed (should not happen with valid init)
    if best_centers is None:
        best_centers = generate_hexagonal_init(0.09)
        best_radii = calculate_radii(best_centers)
        best_sum_r = np.sum(best_radii)

    return best_centers, best_radii, best_sum_r
