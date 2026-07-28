# sol_000001 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4fe936d0) state=633cd7ec sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses multi-start local optimization with SLSQP.
    """
    n_circles = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Helper to check validity
    def is_valid(centers, radii):
        # Boundary checks
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if r < -1e-9: return False
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        # Overlap checks
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    # Objective function: minimize negative sum of radii
    def objective(params):
        # params shape: (n_circles, 3) -> x, y, r
        radii = params[:, 2]
        return -np.sum(radii)

    # Constraints
    # 1. Radii non-negative (handled by bounds usually, but good to be safe)
    # 2. Boundary: r <= x <= 1-r, r <= y <= 1-r
    #    x - r >= 0  => r - x <= 0
    #    1 - x - r >= 0 => r + x - 1 <= 0
    #    y - r >= 0  => r - y <= 0
    #    1 - y - r >= 0 => r + y - 1 <= 0
    # 3. Non-overlap: dist >= r_i + r_j  => (r_i + r_j)^2 <= dist^2
    #    (r_i + r_j)^2 - dist^2 <= 0

    def get_constraints(params):
        constraints = []
        centers = params[:, :2]
        radii = params[:, 2]
        
        # Boundary constraints
        # We can define them as functions returning value <= 0
        # But SLSQP expects type 'ineq' meaning func(x) >= 0.
        # So we negate our <= 0 conditions.
        
        # Boundary: x >= r  => x - r >= 0
        #            x <= 1-r => 1 - x - r >= 0
        #            y >= r  => y - r >= 0
        #            y <= 1-r => 1 - y - r >= 0
        
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 0] - p[i, 2]}) # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i, 0] - p[i, 2]}) # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 1] - p[i, 2]}) # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i, 1] - p[i, 2]}) # 1 - y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 2]}) # r >= 0

        # Non-overlap constraints
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                def overlap_constraint(p, i=i, j=j):
                    c_i = p[i, :2]
                    c_j = p[j, :2]
                    r_i = p[i, 2]
                    r_j = p[j, 2]
                    dist_sq = np.sum((c_i - c_j)**2)
                    return dist_sq - (r_i + r_j)**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    # Actually r can be up to 0.5, but practically smaller.
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n_circles

    # Generate initial guesses
    # Strategy 1: Grid-based perturbation
    # Strategy 2: Random placement
    # Strategy 3: Hexagonal lattice approximation
    
    initial_guesses = []
    
    # 1. Grid-like initialization for 25 circles + 1 random
    # A 5x5 grid has centers at 0.1, 0.3, 0.5, 0.7, 0.9 with r=0.1
    # But we have 26 circles. Let's try to distribute them.
    # Maybe 5 rows of 5 and 1?
    # Let's create a dense grid of 26 points
    # 5x5 is 25. Add one in the center? No, center is occupied.
    # Let's just place them in a grid pattern as best as possible.
    rows = 6
    cols = 5 # 30 spots, we use 26
    # Or 5 rows, 6 cols?
    # Let's try to fit 26 in a grid.
    # 5 rows, roughly 5.2 cols.
    # Let's just randomize positions slightly around a grid.
    
    # Generate a 5x5 grid centers
    coords = []
    for r_idx in range(5):
        for c_idx in range(5):
            coords.append([0.1 + 0.2 * c_idx, 0.1 + 0.2 * r_idx])
    # We have 25 points. Add one more.
    # Where? Maybe center (0.5, 0.5) is taken.
    # Maybe (0.2, 0.2)?
    # Let's just add a random point or shift one.
    # Actually, a hexagonal packing might be better start.
    
    # Hexagonal lattice initialization
    # Try to fit 26 circles in hex pattern
    # Density is higher.
    # Let's try a few random starts which are usually robust enough if the optimizer works.
    
    num_restarts = 15
    
    for _ in range(num_restarts):
        # Random initialization inside [0.1, 0.9] with small radius 0.01
        # This is a safe valid start
        centers_init = np.random.uniform(0.1, 0.9, size=(n_circles, 2))
        radii_init = np.full(n_circles, 0.01)
        
        # Sometimes better to start with larger radii if we know a config exists
        # But 0.01 is safe.
        
        params_init = np.zeros((n_circles, 3))
        params_init[:, :2] = centers_init
        params_init[:, 2] = radii_init
        
        # Try to optimize
        # SLSQP is good but sensitive to start.
        # We can try to relax constraints first? No, SLSQP handles them.
        
        try:
            res = opt.minimize(objective, params_init, method='SLSQP', 
                              bounds=bounds, 
                              constraints=get_constraints(params_init),
                              options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < 0: # If sum radii > 0
                final_centers = res.x[:, :2]
                final_radii = res.x[:, 2]
                
                # Post-process: ensure validity and clean up numerical noise
                # Clip radii to non-negative
                final_radii = np.maximum(final_radii, 0)
                
                # Validate
                if is_valid(final_centers, final_radii):
                    current_sum = np.sum(final_radii)
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = final_centers.copy()
                        best_radii = final_radii.copy()
                        
        except Exception as e:
            # If optimization fails, continue to next restart
            pass

    # If no valid solution found (unlikely with random starts), fallback to a known valid one
    if best_centers is None:
        # Fallback: 26 circles in a loose grid
        # 5x5 grid + 1?
        # Just place 26 circles with small radius
        centers_fb = np.random.uniform(0.15, 0.85, size=(26, 2))
        radii_fb = np.full(26, 0.05)
        # Adjust to make valid
        # Simple grid
        fb_centers = []
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < 26:
                    x = 0.1 + 0.15 * c # spacing 0.15? 5*0.15 = 0.75 + 0.1 = 0.85. ok.
                    y = 0.1 + 0.15 * r
                    fb_centers.append([x, y])
                    idx += 1
        best_centers = np.array(fb_centers)
        best_radii = np.full(26, 0.05) # Radius 0.05, diameter 0.1. Distance between centers 0.15. 0.15 >= 0.1. Valid.
        best_sum_radii = 1.3 # 26 * 0.05

    return best_centers, best_radii, float(best_sum_radii)
