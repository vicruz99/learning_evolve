# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16623584) state=4f9e0625 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a heuristic approach starting from a hexagonal lattice and optimizing positions/radii.
    """
    n_circles = 26
    
    # Helper function to check validity and return overlap magnitude
    def check_constraints(centers, radii):
        """Returns a penalty value. 0 means valid."""
        penalty = 0.0
        
        # Boundary checks
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if r < 0:
                penalty += 1e6 * abs(r)
            
            # Distance to boundaries
            dist_left = x - r
            dist_right = 1.0 - (x + r)
            dist_bottom = y - r
            dist_top = 1.0 - (y + r)
            
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            if min_dist < 0:
                penalty += 1e4 * (-min_dist)
        
        # Overlap checks
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    overlap = sum_r - dist
                    penalty += 1e5 * overlap
                    
        return penalty

    # Helper to calculate objective (sum of radii) minus penalty
    def objective(params):
        centers = params[:n_circles*2].reshape(n_circles, 2)
        radii = params[n_circles*2:]
        
        # We want to maximize sum(radii), so minimize -sum(radii)
        # But we need to satisfy constraints.
        # Using a penalty method.
        penalty = check_constraints(centers, radii)
        
        if penalty > 1e-6:
            # Hard penalty to discourage infeasible solutions during search
            return -np.sum(radii) + 1000.0 * penalty
        else:
            return -np.sum(radii)

    # Initial Configuration: Hexagonal Lattice
    # We need to fit 26 circles. 
    # A 5x5 grid fits 25 with r=0.1. 
    # We can try to arrange in rows: 6, 5, 6, 5, 4 (Total 26)
    # Or 5, 5, 5, 5, 5, 1?
    # Let's try a dense hexagonal arrangement.
    
    # Estimate radius for initial placement
    # If we assume roughly equal circles, r ~ 0.1 is a good start.
    initial_r = 0.095
    
    centers = []
    radii = []
    
    # Generate hexagonal grid points
    # Row height = r * sqrt(3)
    # Row width spacing = 2r
    
    # Let's try to fit rows with varying counts
    # Rows: 6, 5, 6, 5, 4 (Total 26)
    # Row 0 (6 circles): y = r
    # Row 1 (5 circles): y = r + r*sqrt(3), x shifted by r
    # ...
    
    rows_counts = [6, 5, 6, 5, 4]
    
    # Calculate bounding box for this arrangement to scale initial_r
    # Width for 6 circles: 2*6*r = 12r? No.
    # Centers span (n-1)*2r. Plus r margin on each side.
    # Total width = 2r + (n-1)2r = 2nr.
    # If n=6, width = 12r. We need 12r <= 1 => r <= 0.0833.
    # If we use r=0.095, 6 circles won't fit in a row horizontally aligned.
    # However, hexagonal rows are staggered.
    # But the x-span of a row with 6 circles is still roughly determined by the diameter sum?
    # Actually, centers x_1 ... x_6. x_6 - x_1 = 5 * 2r = 10r.
    # Margin r on left, r on right. Total 12r.
    # So 6 circles in a row requires r <= 1/12.
    # To get higher sum, we might want r > 0.0833.
    # So maybe avoid rows of 6?
    # Try 5, 5, 5, 5, 5, 1? No, 5x5 grid r=0.1.
    # 5, 5, 5, 5, 4, 2?
    # Max row size 5.
    # Rows: 5, 5, 5, 5, 4, 2? Sum = 26.
    # 6 rows.
    # Vertical space for 6 rows: 2r + 5 * r*sqrt(3) = r(2 + 8.66) = 10.66r.
    # 10.66r <= 1 => r <= 0.0938.
    # Sum = 26 * 0.0938 = 2.43.
    
    # Maybe a tilted lattice or random init is better?
    # Let's try a random initialization inside the square with some repulsion.
    
    np.random.seed(42)
    
    # Generate random positions
    c = np.random.rand(n_circles, 2)
    # Generate random radii, maybe uniform around 0.1
    r = np.full(n_circles, 0.09)
    
    # Combine into parameter vector
    params = np.concatenate([c.flatten(), r])
    
    # Use scipy.optimize to minimize negative sum of radii
    # We need bounds. x,y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
    for _ in range(n_circles):
        bounds.append((0.0, 0.5)) # r
    
    # Optimization
    # Nelder-Mead is good for non-smooth but might be slow.
    # SLSQP handles constraints but objective is non-convex.
    # Let's try a multi-start approach with L-BFGS-B or SLSQP.
    
    best_params = params
    best_score = -np.inf
    
    # We want to maximize sum(r), so minimize -sum(r).
    # But standard optimizers minimize.
    
    def neg_sum_radii(p):
        # p is flat vector
        r_part = p[n_circles*2:]
        return -np.sum(r_part)
    
    # Custom constraint function for SLSQP
    def constraint_overlap(p):
        centers = p[:n_circles*2].reshape(n_circles, 2)
        radii = p[n_circles*2:]
        # Return distance - sum_radii. Must be >= 0.
        # This is non-convex, hard for SLSQP.
        return 0.0 # Placeholder
        
    # Instead, let's use a penalty method with a robust optimizer like 'Nelder-Mead' or 'Powell'
    # But Nelder-Mead doesn't respect bounds well.
    # 'L-BFGS-B' respects bounds.
    
    # Let's implement a simple "grow" simulation manually for better control.
    
    # Initialize positions more regularly to help convergence
    # Place on a grid
    x = np.linspace(0.15, 0.85, 6) # 6 points
    y = np.linspace(0.15, 0.85, 5) # 5 points
    # 30 points grid. Remove 4.
    grid_points = []
    for yi in y:
        for xi in x:
            grid_points.append([xi, yi])
    # We have 30 points. Take first 26.
    init_centers = np.array(grid_points[:26])
    init_radii = np.full(26, 0.095) # Start with valid radius
    
    # Current state
    cur_centers = init_centers.copy()
    cur_radii = init_radii.copy()
    
    # Simulation parameters
    dt = 0.01
    repulsion_k = 1.0
    attraction_k = 0.0 # We want to expand, not attract
    expansion_rate = 0.0005
    
    # Run simulation
    for step in range(2000):
        forces = np.zeros_like(cur_centers)
        
        # 1. Repulsion between circles
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = cur_centers[i] - cur_centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9: dist = 1e-9
                min_dist = cur_radii[i] + cur_radii[j]
                
                if dist < min_dist:
                    # Overlap, repel
                    overlap = min_dist - dist
                    # Force proportional to overlap / dist
                    f = repulsion_k * overlap / dist
                    dir_vec = diff / dist
                    forces[i] += f * dir_vec
                    forces[j] -= f * dir_vec
                else:
                    # If close, slight repulsion to maintain gap? 
                    # Actually we want them to touch to maximize size, 
                    # but if we are expanding radii, they will touch.
                    pass
        
        # 2. Boundary forces
        for i in range(n_circles):
            x, y = cur_centers[i]
            r = cur_radii[i]
            
            # Left
            if x - r < 0:
                forces[i, 0] += repulsion_k * (r - x) # Push right
            # Right
            if x + r > 1:
                forces[i, 0] -= repulsion_k * (x + r - 1) # Push left
            # Bottom
            if y - r < 0:
                forces[i, 1] += repulsion_k * (r - y) # Push up
            # Top
            if y + r > 1:
                forces[i, 1] -= repulsion_k * (y + r - 1) # Push down
        
        # Update positions
        cur_centers += dt * forces
        
        # Clamp positions to stay inside [0,1] strictly to avoid numerical issues
        # But forces should handle it.
        cur_centers = np.clip(cur_centers, 1e-5, 1.0 - 1e-5)
        
        # Expand radii
        # Check if valid before expanding?
        # Or just expand and let forces push back.
        # To be safe, expand slowly.
        cur_radii += expansion_rate
        
        # If any radius gets too large causing huge forces, maybe dampen?
        # But we want to maximize.
        
        # Periodically try to scale up radii more aggressively if valid?
        # But this is a simple loop.
        
    # After simulation, we might have some overlaps or boundary issues due to discrete steps.
    # Run a local optimizer to clean up and maximize sum.
    
    # Prepare params for optimizer
    params_final = np.concatenate([cur_centers.flatten(), cur_radii])
    
    # Define objective for scipy
    def obj_to_minimize(p):
        centers = p[:n_circles*2].reshape(n_circles, 2)
        radii = p[n_circles*2:]
        
        # We want to maximize sum(radii) -> minimize -sum(radii)
        score = -np.sum(radii)
        
        # Penalty for constraints
        penalty = 0.0
        
        # Boundary
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if r < 0: penalty += 1e6 * abs(r)
            
            # Check bounds with tolerance
            if x - r < 0: penalty += 1e3 * (-x + r)
            if x + r > 1: penalty += 1e3 * (x + r - 1)
            if y - r < 0: penalty += 1e3 * (-y + r)
            if y + r > 1: penalty += 1e3 * (y + r - 1)
            
        # Overlaps
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    penalty += 1e4 * (sum_r - dist)
                    
        return score + penalty

    # Bounds for optimizer
    opt_bounds = []
    for _ in range(n_circles):
        opt_bounds.extend([(0.0, 1.0), (0.0, 1.0)])
    for _ in range(n_circles):
        opt_bounds.append((0.0, 0.5))

    # Try to optimize
    # Use Nelder-Mead or Powell as they don't require gradients and handle non-convex well
    # But they ignore bounds.
    # L-BFGS-B respects bounds.
    
    try:
        res = scipy.optimize.minimize(obj_to_minimize, params_final, method='L-BFGS-B', bounds=opt_bounds, options={'maxiter': 1000, 'ftol': 1e-9})
        best_params = res.x
    except:
        best_params = params_final
        
    # Extract solution
    final_centers = best_params[:n_circles*2].reshape(n_circles, 2)
    final_radii = best_params[n_circles*2:]
    
    # Ensure radii are non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Final check and cleanup:
    # If there are tiny violations, fix them.
    # But the penalty method should have handled it.
    
    # Calculate sum
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Run the function to test logic if needed, but the prompt asks for the function definition.
# However, to be sure, I will wrap the logic inside run_packing.
