# sol_000177 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ae68a5b3) state=9e2094a5 sum of radii=2.567271 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a physics-based optimization approach with unequal radii.
    """
    n = 26
    
    # --- Helper Functions ---
    
    def compute_penalty(centers, radii):
        """
        Computes a penalty value for invalid configurations.
        Positive penalty means invalid.
        """
        penalty = 0.0
        tol = 1e-6
        
        # Boundary constraints
        # Circle i must be inside [0,1]x[0,1]
        # x - r >= 0 => r - x <= 0
        # 1 - x - r >= 0 => r + x - 1 <= 0
        # Same for y
        
        # We accumulate squared violations
        # If valid, penalty contribution is 0.
        
        # Boundary violations
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Left
            if x - r < 0:
                penalty += (x - r) ** 2
            # Right
            if x + r > 1:
                penalty += (x + r - 1) ** 2
            # Bottom
            if y - r < 0:
                penalty += (y - r) ** 2
            # Top
            if y + r > 1:
                penalty += (y + r - 1) ** 2
                
            # Radius non-negativity (should be handled by bounds, but just in case)
            if r < 0:
                penalty += r ** 2

        # Overlap constraints
        # dist(i, j) >= r_i + r_j
        # dist - r_i - r_j >= 0
        
        # Vectorized distance calculation might be faster but loop is fine for N=26
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            for j in range(i + 1, n):
                xj, yj = centers[j]
                rj = radii[j]
                
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = ri + rj - dist
                if overlap > 0:
                    penalty += overlap ** 2
                    
        return penalty

    def objective(x):
        """
        Objective to minimize: -sum(radii) + penalty * lambda
        We want to maximize sum(radii), so we minimize negative sum.
        x contains [x1, y1, r1, x2, y2, r2, ...]
        """
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        for i in range(n):
            centers[i, 0] = x[3*i]
            centers[i, 1] = x[3*i + 1]
            radii[i] = x[3*i + 2]
            
        current_sum_radii = np.sum(radii)
        pen = compute_penalty(centers, radii)
        
        # Penalty weight needs to be high enough to enforce constraints
        # But not so high that it dominates the objective
        # Adaptive penalty or fixed high penalty
        # Let's use a fixed high penalty for simplicity, or scale with iterations if possible
        # However, in a single function call, we can't track iteration easily without closure
        # We will rely on the optimizer navigating the landscape.
        
        # A common trick is to make penalty large.
        # If penalty > 0, we are in invalid region.
        # We want to find valid region with max sum.
        
        return -current_sum_radii + 1000.0 * pen

    # --- Initialization ---
    # Use a hexagonal lattice initialization to start with a dense packing
    centers_init = np.zeros((n, 2))
    radii_init = np.ones(n) * 0.05 # Start with small radius
    
    # Hexagonal grid generation
    # We want to fill the square.
    # Let's try to place points in a pattern.
    # 5 rows of 5 is 25. Add 1 somewhere.
    # Or 6 rows.
    
    # Let's generate a hex grid and pick 26 points
    # Spacing approx 0.2
    points = []
    spacing = 0.22
    row_spacing = spacing * np.sqrt(3) / 2
    
    y = spacing # Start y
    while y < 1.0:
        x = spacing # Start x
        # Determine offset for this row (alternating)
        offset = 0
        row_idx = int((y - spacing) / row_spacing)
        if row_idx % 2 == 1:
            offset = spacing / 2.0
            
        while x < 1.0:
            points.append((x + offset, y))
            x += spacing
        y += row_spacing
        
    # If we have more than 26, trim. If less, add random or duplicate.
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        while len(points) < n:
            points.append((np.random.rand(), np.random.rand()))
            
    # Flatten to x vector
    x0 = []
    for i in range(n):
        if i < len(points):
            px, py = points[i]
        else:
            px, py = 0.5, 0.5
        x0.extend([px, py, 0.05]) # Initial radius small
        
    x0 = np.array(x0)
    
    # --- Optimization ---
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Run optimizer multiple times to avoid local minima
    best_res = None
    best_val = np.inf
    
    # We can try a few random restarts or just one good run.
    # Given the complexity, let's try SLSQP which handles constraints well, 
    # but here we used penalty method. SLSQP can handle explicit constraints too.
    # Let's switch to explicit constraints for better reliability.
    
    # Explicit constraints for SLSQP
    # We need to define constraints as dictionaries
    
    # However, defining 300+ constraints might be slow.
    # Penalty method with a robust optimizer like 'Nelder-Mead' or 'Powell' might be slower but simpler to setup?
    # 'SLSQP' is good for constrained.
    # Let's try to use SLSQP with explicit constraints if possible, but generating them dynamically is hard in the callback.
    # Actually, we can define the constraints in the minimize call.
    
    # Constraints list
    constraints = []
    
    # Boundary constraints for each circle
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # Helper to create constraint dict
    # type 'ineq' means fun(x) >= 0
    
    for i in range(n):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x - r >= 0
        def make_bound_left(idx_x, idx_r):
            return lambda x: x[idx_x] - x[idx_r]
        constraints.append({'type': 'ineq', 'fun': make_bound_left(idx_x, idx_r)})
        
        # 1 - x - r >= 0
        def make_bound_right(idx_x, idx_r):
            return lambda x: 1.0 - x[idx_x] - x[idx_r]
        constraints.append({'type': 'ineq', 'fun': make_bound_right(idx_x, idx_r)})
        
        # y - r >= 0
        def make_bound_bottom(idx_y, idx_r):
            return lambda x: x[idx_y] - x[idx_r]
        constraints.append({'type': 'ineq', 'fun': make_bound_bottom(idx_y, idx_r)})
        
        # 1 - y - r >= 0
        def make_bound_top(idx_y, idx_r):
            return lambda x: 1.0 - x[idx_y] - x[idx_r]
        constraints.append({'type': 'ineq', 'fun': make_bound_top(idx_y, idx_r)})
        
        # Non-negative radius
        # r >= 0 is handled by bounds (0, 0.5)
        
    # Overlap constraints: dist(i,j) - r_i - r_j >= 0
    # This is non-convex, but SLSQP can handle it.
    # There are N*(N-1)/2 = 325 constraints.
    # This might be heavy but let's try.
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i + 1, 3*i + 2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j + 1, 3*j + 2
            
            def make_overlap(idx_xi, idx_yi, idx_ri, idx_xj, idx_yj, idx_rj):
                def fun(x):
                    xi, yi, ri = x[idx_xi], x[idx_yi], x[idx_ri]
                    xj, yj, rj = x[idx_xj], x[idx_yj], x[idx_rj]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - ri - rj
                return fun
            
            constraints.append({'type': 'ineq', 'fun': make_overlap(idx_xi, idx_yi, idx_ri, idx_xj, idx_yj, idx_rj)})

    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    def obj_func(x):
        return -np.sum(x[2::3])

    # Run optimization
    # We might need to restart or use a good initial point.
    # The hex grid initialization should be decent.
    
    # To improve robustness, we can scale the initial radii to be feasible.
    # Check overlap for initial config and shrink radii if needed.
    # But 0.05 is small, should be fine.
    
    try:
        res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            best_res = res
            best_val = res.fun
        else:
            # If failed, maybe try again with different seed?
            # For now, store the result
            best_res = res
            best_val = res.fun
            
    except Exception as e:
        # Fallback to a simple grid packing if optimization fails
        print(f"Optimization failed: {e}")
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.09
        # Just place them in a grid
        k = 0
        for r in range(5):
            for c in range(6): # 5x6=30, take first 26
                if k < n:
                    centers[k] = [0.05 + c*0.18, 0.05 + r*0.18]
                    k += 1
        return centers, radii, np.sum(radii)

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_res.x[3*i]
        centers[i, 1] = best_res.x[3*i + 1]
        radii[i] = best_res.x[3*i + 2]
        
    sum_radii = np.sum(radii)
    
    # Post-processing: validate and clamp if necessary (though optimizer should handle it)
    # Just in case of numerical errors violating constraints slightly
    # We can try to shrink radii slightly to ensure validity
    # But the constraints are inequality >= 0, so valid.
    
    # Check validity
    # Note: The validate_packing function uses 1e-12 tolerance.
    # We should ensure we are well within bounds.
    
    # If the sum is low, maybe the optimizer got stuck.
    # Let's check the sum.
    
    return centers, radii, sum_radii
