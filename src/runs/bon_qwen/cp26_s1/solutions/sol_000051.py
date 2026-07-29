# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 594c2933) state=733178cf sum of radii=1.695284 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
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

def calculate_score(centers, radii):
    """
    Calculates a penalized score. Higher is better.
    Penalties are applied for boundary violations and overlaps.
    """
    n = len(radii)
    score = np.sum(radii)
    
    # Boundary penalties
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Distance to boundaries
        dist_left = x
        dist_right = 1 - x
        dist_bottom = y
        dist_top = 1 - y
        
        min_dist_boundary = min(dist_left, dist_right, dist_bottom, dist_top)
        
        if min_dist_boundary < r:
            violation = r - min_dist_boundary
            score -= 1000 * (violation ** 2)
            
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            req_dist = radii[i] + radii[j]
            
            if dist < req_dist:
                violation = req_dist - dist
                score -= 1000 * (violation ** 2)
                
    return score

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # We will try to optimize for equal radii first, as it's a strong baseline.
    # Variables: x1, y1, x2, y2, ..., xn, yn, r
    # Total variables: 2*n + 1 = 53
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Number of restarts for random search
    num_restarts = 20
    
    for restart in range(num_restarts):
        # Initial guess
        # Place centers in a grid-like pattern to start close to a valid packing
        # But add some noise
        centers = np.zeros((n, 2))
        r_guess = 0.1 # Reasonable starting radius
        
        # Create a grid
        # 5x5 grid is 25, we need 26. 
        # Let's just randomize or use a dense grid with noise
        # A better initial guess is a hexagonal lattice
        
        row_idx = 0
        col_idx = 0
        spacing = 0.22 # Slightly larger than diameter to allow movement
        offset = 0.11
        
        for i in range(n):
            row = i // 5
            col = i % 5
            if row % 2 == 1:
                col_offset = spacing / 2
            else:
                col_offset = 0
            
            x = col * spacing + col_offset + 0.05
            y = row * spacing * math.sqrt(3)/2 + 0.05
            
            if x > 0.95: x = 0.95
            if y > 0.95: y = 0.95
            
            centers[i, 0] = x
            centers[i, 1] = y
        
        # Add random noise
        centers += np.random.uniform(-0.05, 0.05, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial radius
        r_init = 0.09
        
        # Construct initial parameters vector
        # Format: [x1, y1, x2, y2, ..., xn, yn, r]
        x0 = np.zeros(2 * n + 1)
        for i in range(n):
            x0[2*i] = centers[i, 0]
            x0[2*i+1] = centers[i, 1]
        x0[-1] = r_init
        
        # Optimization function to minimize (negative score)
        def objective(params):
            # Extract centers and radius
            c = np.zeros((n, 2))
            for i in range(n):
                c[i, 0] = params[2*i]
                c[i, 1] = params[2*i+1]
            r = params[-1]
            radii = np.full(n, r)
            
            # Clip radius to be positive
            if r < 0: r = 1e-6
            
            return -calculate_score(c, radii)

        # We use a local optimizer. Nelder-Mead is good for non-smooth landscapes.
        # However, with 53 variables it might be slow.
        # Let's try SLSQP with bounds.
        
        bounds = []
        for i in range(2*n):
            bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5)) # Radius bound
        
        try:
            res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Extract result
            final_centers = np.zeros((n, 2))
            for i in range(n):
                final_centers[i, 0] = res.x[2*i]
                final_centers[i, 1] = res.x[2*i+1]
            final_r = res.x[-1]
            final_radii = np.full(n, final_r)
            
            current_score = -res.fun
            
            if current_score > best_score:
                best_score = current_score
                best_centers = final_centers.copy()
                best_radii = final_radii.copy()
                
        except Exception as e:
            continue
            
    # If equal radii optimization didn't reach target, try a local refinement with variable radii
    # But variable radii adds dimensions.
    # Let's just stick to the best equal radii solution found, or try to perturb radii slightly.
    
    # Let's verify the best solution
    if validate_packing(best_centers, best_radii):
        return best_centers, best_radii, np.sum(best_radii)
    else:
        # If validation failed (due to penalty not being strong enough or numerical issues),
        # we might need to scale down radii slightly.
        # Find max valid scale factor
        scale = 1.0
        while not validate_packing(best_centers, best_radii * scale):
            scale -= 0.01
            if scale <= 0: break
        
        return best_centers, best_radii * scale, np.sum(best_radii * scale)
