import numpy as np
import scipy.optimize

def compute_objective(params, n, penalty_weight):
    """
    Objective function to maximize radius r.
    Uses a penalty method to handle boundary and overlap constraints.
    """
    r = params[-1]
    centers = params[:2 * n].reshape(n, 2)
    
    penalty = 0.0
    
    # Boundary constraint violations: circles must stay within [0,1] x [0,1]
    for i in range(n):
        x, y = centers[i]
        if x < r:
            penalty += (r - x) ** 2
        elif x > 1.0 - r:
            penalty += (x - (1.0 - r)) ** 2
            
        if y < r:
            penalty += (r - y) ** 2
        elif y > 1.0 - r:
            penalty += (y - (1.0 - r)) ** 2
            
    # Overlap constraint violations: distance between centers >= 2*r
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = 2.0 * r
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
                
    # We maximize r, so we minimize -r. Large penalty forces feasibility.
    return -r + penalty_weight * penalty

def run_packing():
    n = 26
    
    # Heuristic initialization based on a perturbed hexagonal lattice.
    # Row counts [6, 5, 6, 5, 4] sum to 26 and match known optimal topologies.
    row_counts = [6, 5, 6, 5, 4]
    centers_init = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        y = (row + 0.5) / 5.0
        count = row_counts[row]
        for col in range(count):
            x = (col + 0.5) / count
            # Shift odd rows to approximate hexagonal packing
            if row % 2 == 1:
                x += 1.0 / (2.0 * count)
            # Scale to fit comfortably inside [0,1] initially
            x = 0.15 + 0.7 * np.clip(x, 0.0, 1.0)
            y = 0.15 + 0.7 * y
            centers_init[idx] = [x, y]
            idx += 1
            
    r_init = 0.08
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    # Bounds for positions and radius
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]
    
    # Optimization
    res = scipy.optimize.minimize(
        compute_objective, 
        x0, 
        args=(n, 10000.0), 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 10000, 'ftol': 1e-12}
    )
    
    centers_opt = res.x[:2 * n].reshape(n, 2)
    
    # Determine the largest valid radius for the optimized centers
    min_r = 1.0
    for i in range(n):
        x, y = centers_opt[i]
        min_r = min(min_r, x, 1.0 - x, y, 1.0 - y)
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers_opt[i, 0] - centers_opt[j, 0]
            dy = centers_opt[i, 1] - centers_opt[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_r = min(min_r, dist / 2.0)
            
    # Apply a small safety margin to pass strict numerical validation
    min_r = max(0.0, min_r - 1e-7)
    
    radii_opt = np.full(n, min_r)
    total_sum = np.sum(radii_opt)
    
    return centers_opt, radii_opt, total_sum