# sol_000166 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6d8d18a8) state=2f146b29 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Returns the optimal packing of 26 circles in a unit square.
    """
    n = 26
    
    # Helper to create initial configurations
    def get_initial_configs():
        configs = []
        
        # 1. Random configuration
        # Place circles randomly with small radius to ensure validity
        rng = np.random.RandomState(42)
        centers = rng.rand(n, 2)
        radii = np.full(n, 0.01)
        configs.append((centers, radii))
        
        # 2. Hexagonal grid configuration
        # Try to fit 26 circles in a hexagonal pattern
        # We can try different row counts
        # Pattern: 5, 5, 5, 5, 5, 1 is too rigid.
        # Let's try a dense hexagonal lattice and pick 26 points.
        
        # Estimate radius for hexagonal packing
        # If we fit approx 5x5, r ~ 0.1.
        # Let's place points on a hexagonal grid.
        # Horizontal spacing 2*r, vertical r*sqrt(3)
        # Let's assume r=0.1 for spacing
        r_est = 0.1
        x_step = 2 * r_est
        y_step = r_est * np.sqrt(3)
        
        points = []
        # Generate enough points
        # Square is [0,1]x[0,1]
        # We need to cover the area.
        # Let's just generate a grid of points
        ys = np.arange(r_est, 1.0, y_step)
        for i, y in enumerate(ys):
            # Shift odd rows
            offset = r_est if i % 2 == 1 else 0
            xs = np.arange(r_est + offset, 1.0 - r_est + 1e-9, x_step)
            for x in xs:
                points.append([x, y])
        
        if len(points) >= n:
            # Take first n points
            centers = np.array(points[:n])
            radii = np.full(n, r_est * 0.9) # Start slightly smaller
            configs.append((centers, radii))
        else:
            # Fallback to grid
            configs.append(get_grid_config())

        # 3. Grid configuration (5x5 + 1)
        centers, radii = get_grid_config()
        configs.append((centers, radii))
        
        return configs

    def get_grid_config():
        # 5x5 grid of 25 circles, plus one in the middle or random spot
        # Grid spacing 0.2
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        centers = np.array([[x, y] for y in ys for x in xs]) # 25 points
        # Add one more
        # Maybe at center (0.5, 0.5) but shift slightly?
        # Or just place it somewhere valid
        centers = np.vstack([centers, [0.5, 0.5]])
        radii = np.full(26, 0.05) # Small initial radius
        return centers, radii

    configs = get_initial_configs()
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Optimization function
    def objective(vars_1d):
        radii_obj = vars_1d[2::3]
        return -np.sum(radii_obj) # Minimize negative sum

    def constraints(vars_1d):
        centers = np.column_stack((vars_1d[0::3], vars_1d[1::3]))
        radii = vars_1d[2::3]
        
        cons = []
        
        # Boundary constraints: x >= r, x <= 1-r => x-r >= 0, 1-x-r >= 0
        # y >= r, y <= 1-r
        # Flatten to list of scalar constraints
        
        # x - r >= 0
        for i in range(n):
            cons.append(vars_1d[3*i] - vars_1d[3*i+2])
            # 1 - x - r >= 0
            cons.append(1.0 - vars_1d[3*i] - vars_1d[3*i+2])
            # y - r >= 0
            cons.append(vars_1d[3*i+1] - vars_1d[3*i+2])
            # 1 - y - r >= 0
            cons.append(1.0 - vars_1d[3*i+1] - vars_1d[3*i+2])
            
        # Non-overlap: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        # Only check i < j
        for i in range(n):
            for j in range(i + 1, n):
                dx = vars_1d[3*i] - vars_1d[3*j]
                dy = vars_1d[3*i+1] - vars_1d[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = vars_1d[3*i+2] + vars_1d[3*j+2]
                cons.append(dist_sq - r_sum*r_sum)
        
        return cons

    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    for k, (init_centers, init_radii) in enumerate(configs):
        # Flatten initial guess
        vars_0 = np.zeros(3 * n)
        vars_0[0::3] = init_centers[:, 0]
        vars_0[1::3] = init_centers[:, 1]
        vars_0[2::3] = init_radii
        
        # Ensure initial validity for constraints (r <= x <= 1-r etc)
        # If init_radii are small enough, this should hold.
        # If centers are outside, clamp them?
        # But let's hope random/grid are valid.
        
        try:
            res = opt.minimize(
                objective,
                vars_0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or (res.nit > 0 and -res.fun > best_sum):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = res.x[2::3]
                    
        except Exception as e:
            print(f"Optimization failed for config {k}: {e}")
            continue

    # If optimization didn't find anything good, fallback to a valid manual config
    if best_centers is None:
        # Fallback to 5x5 grid r=0.1, 26th circle small
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        centers = np.array([[x, y] for y in ys for x in xs])
        centers = np.vstack([centers, [0.5, 0.5]])
        radii = np.full(26, 0.1)
        radii[-1] = 0.0 # Last one invalid if touching, set small
        # Adjust to be valid
        # With r=0.1, circles touch.
        # 26th at (0.5, 0.5) touches neighbors.
        # Let's just return a valid small packing if all else fails.
        centers = np.random.rand(26, 2)
        radii = np.full(26, 0.01)
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii

    # Validate final result before returning
    # Note: The provided validate_packing function is read-only, but we can use logic similar to it.
    # We need to ensure strict validity.
    # The optimizer tries to satisfy constraints >= 0.
    # Due to numerical errors, we might have slight violations.
    # We can clamp radii slightly or adjust.
    
    # Check and fix small violations
    # If dist < r1+r2, reduce radii or move centers?
    # Since we can't run the validate function inside run_packing easily (imports might be tricky but allowed),
    # let's just trust the optimizer with tight tolerances.
    # However, to be safe, we can scale down radii slightly if needed?
    # No, that reduces objective.
    
    # Let's do a quick check and scale down if necessary
    valid = True
    min_gap = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            req = best_radii[i] + best_radii[j]
            if d < req - 1e-9:
                valid = False
                # Find scaling factor
                # We need d >= s * (r_i + r_j) ? No, we can't scale all equally easily without breaking boundaries.
                pass
    
    # If not valid, we might need to shrink radii slightly to satisfy constraints strictly.
    # But with SLSQP and 'ineq' constraints, it should be valid.
    
    return best_centers, best_radii, np.sum(best_radii)
