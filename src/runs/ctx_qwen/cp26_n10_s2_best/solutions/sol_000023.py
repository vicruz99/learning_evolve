# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d17cbe8) state=10ebc8b6 sum of radii=2.044837 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # Penalty weight for constraint violations
    # High value to ensure constraints are respected
    penalty_weight = 5000.0
    
    def objective(x):
        # Unpack variables
        # x[0:n*2] are centers (x, y) for each circle
        # x[n*2:] are radii
        centers = x[:n*2].reshape(n, 2)
        radii = x[n*2:]
        
        sum_radii = np.sum(radii)
        penalty = 0.0
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # Equivalent to: r - x <= 0, x + r - 1 <= 0, etc.
        # We penalize positive violations squared.
        
        # Left boundary: x >= r  => r - x <= 0
        v_left = radii - centers[:, 0]
        # Right boundary: x <= 1-r => x + r - 1 <= 0
        v_right = centers[:, 0] + radii - 1.0
        # Bottom boundary: y >= r => r - y <= 0
        v_bottom = radii - centers[:, 1]
        # Top boundary: y <= 1-r => y + r - 1 <= 0
        v_top = centers[:, 1] + radii - 1.0
        
        # Accumulate squared positive violations
        v_left = np.maximum(0, v_left)
        v_right = np.maximum(0, v_right)
        v_bottom = np.maximum(0, v_bottom)
        v_top = np.maximum(0, v_top)
        
        penalty += np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Overlap constraints: dist(i, j) >= r_i + r_j
        # Violation: r_i + r_j - dist > 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    penalty += overlap**2
                    
        return -sum_radii + penalty_weight * penalty

    def get_bounds():
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
        return bounds

    bounds = get_bounds()
    
    best_result = None
    best_obj = np.inf
    
    # Helper to generate initial configurations
    def get_init_configurations():
        configs = []
        
        # 1. Random initialization with small radii
        for _ in range(3):
            centers = np.random.rand(n, 2)
            radii = np.full(n, 0.02) # Start small to avoid immediate penalty
            x0 = np.concatenate([centers.flatten(), radii])
            configs.append(x0)
            
        # 2. Grid initialization (5x5 is 25, need 26)
        # Place 25 in 5x5 grid, 1 in center? 
        # Actually, let's use a perturbed grid.
        # 5x5 grid points: 0.1, 0.3, 0.5, 0.7, 0.9
        # This fits radius 0.1 perfectly. 
        # We can try to start there with slightly smaller radius and see if optimizer expands.
        # But 26th circle? Let's add it at (0.0, 0.0) or center?
        # Center is occupied.
        # Let's just use a dense random sample or hex grid.
        pass

        # 3. Hexagonal packing initialization
        # Try to pack circles of radius r ~ 0.09 in hexagonal lattice
        r_init = 0.08
        y = r_init
        row = 0
        points = []
        while y + r_init <= 1.0:
            # Row offset
            if row % 2 == 0:
                x_start = r_init
                dx = 2 * r_init
            else:
                x_start = r_init + r_init # shift by r? No, shift by r for hex?
                # In hex packing, horizontal shift is r. 
                # If row 0 centers at r, 3r, 5r...
                # Row 1 centers at 2r, 4r, 6r...
                # Wait, distance between (r, y0) and (2r, y1) is sqrt(r^2 + (sqrt(3)r)^2) = 2r.
                # Yes, shift is r.
                x_start = 2 * r_init
                dx = 2 * r_init
            
            x = x_start
            while x + r_init <= 1.0:
                if len(points) < n:
                    points.append([x, y])
                x += dx
            y += math.sqrt(3) * r_init
            row += 1
        
        if len(points) >= n:
            centers = np.array(points[:n])
            radii = np.full(n, r_init)
            # Normalize radii? No, keep fixed initial.
            # But optimizer might change them.
            x0 = np.concatenate([centers.flatten(), radii])
            configs.append(x0)
        else:
            # Fallback to random if hex doesn't fill enough (unlikely)
            centers = np.random.rand(n, 2)
            radii = np.full(n, 0.02)
            x0 = np.concatenate([centers.flatten(), radii])
            configs.append(x0)

        # 4. Uniform grid 6x5 = 30 points, pick 26? 
        # Just random perturbation of a dense set.
        # 5x6 grid
        xs = np.linspace(0.1, 0.9, 6) # 6 points
        ys = np.linspace(0.1, 0.9, 5) # 5 points
        # Total 30 points.
        grid_points = []
        for y_val in ys:
            for x_val in xs:
                grid_points.append([x_val, y_val])
        # Shuffle and pick 26
        np.random.shuffle(grid_points)
        centers = np.array(grid_points[:n])
        radii = np.full(n, 0.02)
        x0 = np.concatenate([centers.flatten(), radii])
        configs.append(x0)

        return configs

    init_configs = get_init_configurations()
    
    for x0 in init_configs:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res
        except Exception as e:
            print(f"Optimization error: {e}")
            continue
            
    # Extract best solution
    if best_result is None:
        # Fallback
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.01)
    else:
        x_opt = best_result.x
        centers = x_opt[:n*2].reshape(n, 2)
        radii = x_opt[n*2:]
        
    # Validate and potentially fix small violations due to numerical precision
    # The penalty method might leave tiny violations if penalty_weight isn't high enough
    # or if the solver stops early. 
    # Let's do a check and reduce radii slightly if needed.
    
    # Check overlaps
    valid = True
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            # Fix radius
            r = min(x, 1-x, y, 1-y)
            radii[i] = max(0, r)
            
    # Check overlaps and fix by reducing radius of smaller circle
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            req_dist = radii[i] + radii[j]
            if dist < req_dist - 1e-7:
                # Overlap detected. Reduce radii.
                # Simple strategy: scale both down proportionally to fit distance
                # Or just reduce the smaller one?
                # To preserve sum, maybe scale both.
                # New sum r_i' + r_j' = dist
                # Let's maintain ratio or just set sum = dist
                current_sum = radii[i] + radii[j]
                if current_sum > 0:
                    factor = dist / current_sum
                    radii[i] *= factor
                    radii[j] *= factor
                else:
                    radii[i] = dist / 2
                    radii[j] = dist / 2

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Final sanity check
if __name__ == "__main__":
    centers, radii, s = run_packing()
    # Run validation logic locally to ensure it passes
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any(): return False
        for i in range(n):
            if radii[i] < 0: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12: return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12: return False
        return True

    if validate_packing(centers, radii):
        print(f"Valid packing found. Sum of radii: {s}")
    else:
        print("Validation failed, but solution returned.")
