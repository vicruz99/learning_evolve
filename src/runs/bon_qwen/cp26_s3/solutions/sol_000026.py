# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfc1b343) state=76529ecb sum of radii=0.846340 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Generate initial positions based on a hexagonal lattice
    centers = []
    row = 0
    count = 0
    # Hexagonal packing parameters
    # We estimate a radius of ~0.105 for 26 circles.
    # Diameter ~ 0.21. 
    # Vertical spacing in hex packing: sqrt(3)/2 * diameter ~ 0.18
    # We can fit roughly 6 rows.
    
    r_init = 0.10
    dx = 2.0 * r_init
    dy = np.sqrt(3) * r_init
    
    y = r_init
    while count < n:
        row_type = 0 if (row % 2 == 0) else 1
        
        # In even rows, start at x = r_init. In odd rows, start at x = r_init + r_init
        x_start = r_init if row_type == 0 else 2 * r_init
        x = x_start
        
        while x + r_init <= 1.0 + 1e-9:
            if count < n:
                centers.append([x, y])
                count += 1
            x += dx
            
        y += dy
        row += 1
        
    centers = np.array(centers)
    # If we generated more, trim (though loop should prevent this)
    if centers.shape[0] > n:
        centers = centers[:n]
        
    # Normalize/Scale logic is handled in the optimization objective, 
    # but we initialize radii based on the current fit.
    # Actually, we will treat positions as variables and scale them.
    # For the optimizer, it's easier to keep radii fixed relative to positions 
    # or just optimize positions and find max min-distance.
    # However, to maximize sum of radii, we can just scale the whole system.
    # Let's optimize positions of 26 points in [0,1]^2 to maximize 
    # the minimum distance between any pair and to boundaries.
    # Then r = min_dist / 2.
    # Sum of radii = n * r.
    
    def objective(params):
        # params contains 2*n coordinates
        pts = params.reshape((n, 2))
        min_dist = 1.0
        
        # Distance to boundaries
        for i in range(n):
            x, y = pts[i]
            min_dist = min(min_dist, x, 1 - x, y, 1 - y)
            
        # Distance between circles
        # We only need to check a subset or use vectorization for speed
        # For n=26, O(n^2) is ~338 checks, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((pts[i] - pts[j])**2))
                min_dist = min(min_dist, dist)
                
        return -min_dist # Minimize negative distance = Maximize distance

    # Initial guess
    x0 = centers.flatten()
    
    # Bounds for coordinates [0, 1]
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Optimization
    # Use Nelder-Mead or Powell. Nelder-Mead is derivative-free and robust for this.
    # We run a few random restarts to avoid local minima if necessary, 
    # but hexagonal start is very close to optimal.
    
    res = minimize(objective, x0, method='Nelder-Mead', 
                   bounds=bounds, options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if not res.success:
        # Fallback or simple warning, but we proceed with best found
        pass
        
    optimal_centers = res.x.reshape((n, 2))
    
    # Calculate max radius based on optimal positions
    min_dist = 1.0
    
    for i in range(n):
        x, y = optimal_centers[i]
        min_dist = min(min_dist, x, 1 - x, y, 1 - y)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((optimal_centers[i] - optimal_centers[j])**2))
            min_dist = min(min_dist, dist)
            
    optimal_radius = min_dist / 2.0
    radii = np.full(n, optimal_radius)
    
    # Verify and return
    # The problem asks to maximize sum of radii. 
    # With equal radii, this is equivalent to maximizing the single radius.
    
    sum_radii = np.sum(radii)
    
    return optimal_centers, radii, sum_radii

# Helper to validate locally if needed, but the function signature is strict.
