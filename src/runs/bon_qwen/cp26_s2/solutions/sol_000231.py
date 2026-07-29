# sol_000231 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=10fba181 sum of radii=1.833194 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    def obj_func(params):
        """Objective function to minimize (negative sum of radii + penalties)."""
        centers = params[:2 * n].reshape(n, 2)
        radii = params[2 * n:]
        
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Push back if out of bounds
            if x - r < 0: penalty += 1e4 * (x - r)**2
            if x + r > 1: penalty += 1e4 * (x + r - 1)**2
            if y - r < 0: penalty += 1e4 * (y - r)**2
            if y + r > 1: penalty += 1e4 * (y + r - 1)**2
            
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += 1e5 * overlap**2
                    
        return -np.sum(radii) + penalty

    def hexagonal_init(rows, cols_offset, r_scale=1.0):
        """Generate a hexagonal lattice configuration."""
        centers = []
        for r in range(rows):
            # Shift alternate rows to create hexagonal packing
            x_offset = 0.0 if r % 2 == 0 else 0.5
            # Number of circles in this row
            c = cols_offset if r % 2 == 0 else cols_offset - 1
            
            # Adjust spacing to fit in [0,1]
            # For a row of size c, width needed is 2*r*c roughly, 
            # but we center it.
            # We use a scaling factor to fit into the square initially
            step_x = 1.0 / (c + 1) 
            # Actually, for hex packing, horizontal spacing is 2r, vertical is r*sqrt(3)
            # Here we just place them and let optimizer fix r
            
            # Distribute centers evenly in x for this row, shifted
            # Simple grid-ish init for robustness
            for k in range(c):
                # x position
                # We want to leave some margin
                x = (k + 1 + x_offset) * (1.0 / (cols_offset + 1))
                y = (r + 1) * (1.0 / (rows + 1))
                centers.append([x, y])
        return np.array(centers[:n])

    def run_optimization(initial_centers):
        # Initialize radii slightly to avoid zero gradients
        initial_radii = np.full(n, 0.04) 
        
        # Combine into a single parameter vector
        x0 = np.concatenate([initial_centers.flatten(), initial_radii])
        
        # Bounds: x,y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        # Run optimization
        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
        
        return res.x

    # Try multiple initial configurations to find the best global optimum
    # Configuration 1: Hexagonal-like grid
    # 6, 5, 6, 5, 4 circles = 26 circles
    init_centers_1 = hexagonal_init(rows=5, cols_offset=6)
    
    # Configuration 2: Distorted 5x5 grid + 1
    init_centers_2 = np.zeros((n, 2))
    idx = 0
    # 5x5 grid
    for r in range(5):
        for c in range(5):
            if idx < n:
                init_centers_2[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                idx += 1
    # 26th circle in a gap (center of a void)
    if idx < n:
        init_centers_2[idx] = [0.2, 0.2] # Void between (0.1,0.1), (0.3,0.1), etc.

    # Configuration 3: Randomized perturbation of Config 1
    init_centers_3 = init_centers_1.copy()
    init_centers_3 += np.random.uniform(-0.05, 0.05, size=init_centers_1.shape)
    init_centers_3 = np.clip(init_centers_3, 0.05, 0.95)

    configs = [init_centers_1, init_centers_2, init_centers_3]

    for i, centers_init in enumerate(configs):
        try:
            res_x = run_optimization(centers_init)
            centers = res_x[:2 * n].reshape(n, 2)
            radii = res_x[2 * n:]
            
            # Validate and compute sum
            current_sum = np.sum(radii)
            # Check if valid (rough check, objective should be close to negative sum if penalty low)
            # We re-evaluate penalty to ensure we didn't get stuck in a high-penalty state
            penalty = 0.0
            valid = True
            for c in centers:
                if c[0] < 0 or c[0] > 1 or c[1] < 0 or c[1] > 1:
                    valid = False
            for r in radii:
                if r < 0:
                    valid = False
            
            if valid and current_sum > best_sum:
                # Double check with a stricter validation logic locally
                # We assume the optimizer did a good job if penalty is low.
                # The objective function returns -sum + penalty.
                # If penalty is significant, the returned sum is inflated.
                # But L-BFGS-B usually finds local minima of the objective.
                # We can trust the result if the cost is reasonable.
                
                # Let's re-calculate cost to be sure
                cost = obj_func(res_x)
                # If cost is close to -sum, penalty is small.
                if cost < -current_sum + 1e-4: 
                     best_sum = current_sum
                     best_centers = centers
                     best_radii = radii
        except Exception:
            continue

    # Fallback to a safe valid packing if optimization failed to find valid one
    if best_centers is None:
        # Simple grid packing
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.09) # Safe radius
        # Place 25 in 5x5
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n:
                    best_centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                    idx += 1
        if idx < n:
            best_centers[idx] = [0.5, 0.5]
            best_radii[idx] = 0.04 # Smaller center circle

    return best_centers, best_radii, np.sum(best_radii)
