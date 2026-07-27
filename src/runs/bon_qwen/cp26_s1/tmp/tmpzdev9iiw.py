import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import itertools

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # Helper to generate initial centers in a hexagonal grid
    def get_initial_centers(n):
        # Try to pack n circles in a hexagonal pattern
        # Estimate grid size
        # For hex packing, density is high. 
        # Let's just place them on a grid and let the optimizer move them.
        # A 5x5 grid has 25 points. We need 26.
        # Let's try a 5x5 grid plus one extra, or a denser grid.
        # Simple approach: fill a grid with spacing 0.2
        centers = []
        y = 0.1
        while len(centers) < n:
            x = 0.1
            row_len = 0
            while x <= 0.9 and len(centers) < n:
                centers.append([x, y])
                x += 0.2
                row_len += 1
            y += 0.18 # Approx sqrt(3)/2 * 0.2
        return np.array(centers[:n])

    # Initial guess
    initial_centers = get_initial_centers(n_circles)
    initial_radii = np.full(n_circles, 0.05) # Small radius
    
    # Flatten variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    # Actually easier to keep structure [x, y, r] blocks or flat
    # Let's use flat array of size 3*n
    # Order: x0, y0, r0, x1, y1, r1 ...
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = initial_centers[i, 0]
        x0[3*i + 1] = initial_centers[i, 1]
        x0[3*i + 2] = initial_radii[i]

    # Bounds for each variable
    # x in [0, 1], y in [0, 1], r in [0, 1] (actually r <= 0.5)
    bounds = []
    for i in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Objective function: minimize -sum(r)
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    # Gradient of objective
    def objective_grad(vars):
        grad = np.zeros_like(vars)
        # Derivative w.r.t r is -1
        grad[2::3] = -1.0
        return grad

    # Constraints
    # 1. Wall constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # Inequality form: fun(vars) >= 0
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    def wall_constraints(vars):
        res = np.zeros(4 * n_circles)
        for i in range(n_circles):
            idx = 3 * i
            x = vars[idx]
            y = vars[idx+1]
            r = vars[idx+2]
            
            res[4*i]     = x - r        # x >= r
            res[4*i + 1] = 1.0 - x - r  # 1-x >= r
            res[4*i + 2] = y - r        # y >= r
            res[4*i + 3] = 1.0 - y - r  # 1-y >= r
        return res

    def wall_constraints_jac(vars):
        j = np.zeros((4 * n_circles, 3 * n_circles))
        for i in range(n_circles):
            idx = 3 * i
            # Constraint: x - r >= 0
            j[4*i, idx] = 1.0
            j[4*i, idx+2] = -1.0
            
            # Constraint: 1 - x - r >= 0
            j[4*i+1, idx] = -1.0
            j[4*i+1, idx+2] = -1.0
            
            # Constraint: y - r >= 0
            j[4*i+2, idx+1] = 1.0
            j[4*i+2, idx+2] = -1.0
            
            # Constraint: 1 - y - r >= 0
            j[4*i+3, idx+1] = -1.0
            j[4*i+3, idx+2] = -1.0
        return j

    # 2. Pairwise non-overlap constraints
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    pairs = list(itertools.combinations(range(n_circles), 2))
    n_pairs = len(pairs)

    def pair_constraints(vars):
        res = np.zeros(n_pairs)
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        for k, (i, j) in enumerate(pairs):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dr = rs[i] + rs[j]
            res[k] = dx*dx + dy*dy - dr*dr
        return res

    def pair_constraints_jac(vars):
        j = np.zeros((n_pairs, 3 * n_circles))
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        for k, (i, j) in enumerate(pairs):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dr = rs[i] + rs[j]
            
            # d/dxi (dx^2) = 2*dx
            j[k, 3*i] = 2.0 * dx
            # d/dxj (dx^2) = 2*dx * (-1) = -2*dx
            j[k, 3*j] = -2.0 * dx
            
            # d/dyi (dy^2) = 2*dy
            j[k, 3*i + 1] = 2.0 * dy
            # d/dyj (dy^2) = -2*dy
            j[k, 3*j + 1] = -2.0 * dy
            
            # d/dri (-dr^2) = -2*dr * 1 = -2*dr
            j[k, 3*i + 2] = -2.0 * dr
            # d/drj (-dr^2) = -2*dr * 1 = -2*dr
            j[k, 3*j + 2] = -2.0 * dr
            
        return j

    # Define constraint objects
    # Lower bound 0 for all inequality constraints
    wall_con = NonlinearConstraint(wall_constraints, 0, np.inf, jac=wall_constraints_jac)
    pair_con = NonlinearConstraint(pair_constraints, 0, np.inf, jac=pair_constraints_jac)
    
    constraints = [wall_con, pair_con]
    
    # Optimization
    best_res = None
    best_sum_r = -np.inf
    
    # Run multiple times with slight perturbations to avoid local minima
    for trial in range(5):
        # Perturb initial centers slightly
        noise = np.random.normal(0, 0.01, (n_circles, 2))
        # Ensure inside bounds after noise
        centers_perturbed = np.clip(initial_centers + noise, 0.05, 0.95)
        # Also maybe randomize radii slightly
        radii_perturbed = np.clip(initial_radii + np.random.normal(0, 0.005, n_circles), 0.001, 0.2)
        
        x0_perturbed = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0_perturbed[3*i] = centers_perturbed[i, 0]
            x0_perturbed[3*i+1] = centers_perturbed[i, 1]
            x0_perturbed[3*i+2] = radii_perturbed[i]
            
        # Check validity of start point (should be valid with small radii)
        # If not, reduce radii
        # Simple check: if any constraint violated significantly, reduce r
        # But with small r, it should be fine.
        
        try:
            res = minimize(objective, x0_perturbed, method='trust-constr', 
                           jac=objective_grad, bounds=bounds, constraints=constraints,
                           options={'verbose': 0, 'maxiter': 500})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_res = res
        except Exception as e:
            print(f"Optimization trial {trial} failed: {e}")
            continue

    if best_res is None:
        # Fallback to the first attempt result or a simple grid
        # Just run once without perturbations if all failed? 
        # Or return the last x0
        print("Optimization failed to find a solution, returning initial guess.")
        # This shouldn't happen with valid start
        best_res = res 
        
    # Extract solution
    final_vars = best_res.x
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i, 0] = final_vars[3*i]
        centers[i, 1] = final_vars[3*i+1]
        radii[i] = final_vars[3*i+2]
        
    # Validate and fix tiny negative radii or out of bounds due to float errors
    radii = np.maximum(radii, 0.0)
    # Clamp centers
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    # Re-adjust radii if clamping made them invalid (though clip ensures valid)
    # Actually clip ensures center is valid for given radius, but radius might be too big for new center?
    # No, if center was valid, and we clamp center towards center, distance to wall increases?
    # Wait, if center was 0.1 and r=0.1, valid. Clip 0.1 with [0.1, 0.9] -> 0.1. OK.
    # If center was 0.05 and r=0.1 (invalid), clip -> 0.1. Now dist to 0 is 0.1. OK.
    # But if center was 0.05 and r=0.1, and we clamp to 0.1, it's valid.
    # However, if center was 0.95 and r=0.1, valid. Clip -> 0.95. OK.
    # The issue is if center was 0.05, r=0.06 (invalid). Clip x to 0.06. Now x-r = 0. Valid.
    # But wait, if we have multiple circles, moving one might cause overlap.
    # But the optimizer should have found a valid solution.
    # Just ensure radii are valid w.r.t boundaries for the returned centers.
    for i in range(n_circles):
        c = centers[i]
        r = radii[i]
        max_r = min(c[0], 1-c[0], c[1], 1-c[1])
        if r > max_r + 1e-9:
            radii[i] = max_r

    return centers, radii, np.sum(radii)