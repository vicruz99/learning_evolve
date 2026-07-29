# sol_000273 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=00dbb4fc sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization followed by numerical optimization.
    """
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    # This pattern is denser than a square grid and helps escape poor local optima.
    centers = np.zeros((n, 2))
    
    # Parameters for hexagonal packing
    # We pack points with initial spacing slightly less than 1 to allow expansion
    r_init = 0.09 
    dx = 2 * r_init
    dy = r_init * math.sqrt(3)
    
    idx = 0
    row = 0
    while idx < n:
        y = r_init + row * dy
        x_offset = (r_init * 0.5) if (row % 2 == 1) else 0
        x = x_offset
        
        while x < 1.0 - r_init + 1e-6 and idx < n:
            centers[idx, 0] = x + r_init
            centers[idx, 1] = y
            idx += 1
            x += dx
        row += 1
        
    # Trim to exactly n circles if we generated more
    centers = centers[:n]
    
    # 2. Define the objective function and constraints for optimization
    # We want to maximize r such that:
    # dist(ci, cj) >= 2r
    # boundary distance >= r
    # 
    # Equivalent to: maximize min( min_ij(dist(ci, cj)/2), min_i(min(xi, 1-xi, yi, 1-yi)) )
    # 
    # We will use a "smooth" min approximation or simply optimize the positions 
    # to maximize the minimum pairwise distance and boundary distance.
    # Since scipy doesn't handle 'min' well in gradients, we can minimize the 
    # negative of the smallest distance.
    
    def objective(vars):
        # vars is flattened array of x, y coordinates
        c = vars.reshape((n, 2))
        
        min_dist = float('inf')
        
        # Check pairwise distances
        # We can optimize this by only checking neighbors, but for n=26, O(n^2) is fine.
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                if d < min_dist:
                    min_dist = d
        
        # Check boundary distances
        # Distance to left/right boundaries
        dist_x = np.minimum(c[:, 0], 1.0 - c[:, 0])
        dist_y = np.minimum(c[:, 1], 1.0 - c[:, 1])
        min_bound = np.minimum(dist_x, dist_y).min()
        
        # The feasible radius is limited by the smallest of these
        min_val = min(min_dist / 2.0, min_bound)
        
        # We want to maximize this min_val, so minimize negative
        return -min_val

    # Bounds for centers: [0, 1]
    bounds = [(0, 1)] * (2 * n)
    
    # 3. Run optimization
    # We use L-BFGS-B which handles bounds well.
    # Since the objective has a 'min' function, gradients are discontinuous.
    # We might need multiple restarts or a method robust to this.
    # However, starting from a good lattice usually lands in the basin of attraction.
    
    # To improve robustness, we can run a few iterations with a "repulsive" force simulation first?
    # Or just rely on scipy. Let's try scipy with a smooth approximation if needed, 
    # but standard minimize might get stuck.
    
    # Let's use a 'SLSQP' or 'L-BFGS-B'. L-BFGS-B is good for bounds.
    # The objective is non-smooth. 
    # A common trick is to maximize the sum of reciprocals of distances or similar, 
    # but minimizing the negative of the minimum distance is the direct goal.
    
    # To handle the non-smoothness, we can use a technique where we optimize for a fixed r
    # and increase r, but that's complex to code robustly.
    # Instead, let's use a simple gradient-free or robust solver if available, 
    # or just L-BFGS-B with a smoothing parameter.
    
    # Let's try a simple smoothing: 
    # max(x, y) ~ log(exp(alpha*x) + exp(alpha*y)) / alpha
    # min(x, y) ~ -max(-x, -y)
    # We want to maximize min(d_ij/2, b_i).
    # Let's approximate min(S) where S is a list of distances.
    # Softmin(S) = -1/alpha * log( sum(exp(-alpha * s)) )
    
    alpha = 50.0 # Smoothing parameter
    
    def smooth_objective(vars):
        c = vars.reshape((n, 2))
        
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                dists.append(d / 2.0)
        
        for i in range(n):
            dists.append(c[i, 0])
            dists.append(1.0 - c[i, 0])
            dists.append(c[i, 1])
            dists.append(1.0 - c[i, 1])
            
        dists = np.array(dists)
        
        # Soft minimum
        # To avoid overflow, subtract max
        m = np.max(dists)
        soft_min = m - (1.0/alpha) * np.log(np.sum(np.exp(alpha * (dists - m))))
        
        return -soft_min

    # Initial guess from hexagonal pattern
    x0 = centers.flatten()
    
    # Optimize
    res = minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-12})
    
    # If the optimizer gets stuck or the smoothing distorts too much, 
    # we can try to refine with the exact objective using the result as start.
    # But L-BFGS-B on smooth approx is usually very effective for packing.
    
    best_centers = res.x.reshape((n, 2))
    
    # 4. Calculate the actual radius based on the optimized positions
    # The radius is the minimum distance to any other circle or boundary.
    min_dist = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            if d < min_dist:
                min_dist = d
        
    # Boundary distances
    for i in range(n):
        d_bound = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                      best_centers[i, 1], 1.0 - best_centers[i, 1])
        if d_bound < min_dist:
            min_dist = d_bound
            
    radius = min_dist / 2.0
    
    # Create radii array
    radii = np.full(n, radius)
    sum_radii = np.sum(radii)
    
    # Validate
    # (Implicitly valid by construction, but good to be sure)
    # The optimization ensures distances >= 2*radius approximately.
    
    return best_centers, radii, sum_radii
