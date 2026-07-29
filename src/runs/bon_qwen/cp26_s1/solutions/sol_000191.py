# sol_000191 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 320c78c6) state=5b8e9e4e sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # --- Initialization ---
    # We start with a hexagonal-like arrangement to provide a good initial guess.
    # We aim for roughly 5-6 rows.
    # Let's try to fit 26 circles.
    # A 5x5 grid is 25 circles. Hexagonal packing allows more density.
    # Let's generate centers based on a hexagonal lattice with a tentative radius.
    
    # Tentative radius for initialization
    r_init = 0.09
    
    centers = []
    # Try to fill rows
    # Row y-coordinates: r, r + sqrt(3)r, ...
    # Row x-coordinates: r, r + 2r, ... (shifted by r for odd rows)
    
    row_idx = 0
    while len(centers) < n:
        y = r_init + row_idx * (math.sqrt(3) * r_init)
        
        # Determine shift for this row (hexagonal shift)
        # Even rows (0, 2, ...): start at r
        # Odd rows (1, 3, ...): start at 2r (shifted right by r)
        if row_idx % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
        
        x = x_start
        row_circles = 0
        while x <= 1 - r_init and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_init
            row_circles += 1
        
        row_idx += 1
        
    centers = np.array(centers[:n])
    radii = np.full(n, r_init)
    
    # --- Optimization ---
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 3 * n variables.
    
    # Initial vector
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: 0 <= x <= 1, 0 <= y <= 1, 0 <= r (upper bound not strictly needed but good for stability)
    # However, r can be up to 0.5.
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    def objective(z):
        return -np.sum(z[2::3]) # sum of radii
        
    # Constraints
    # 1. Boundary constraints:
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => y + r - 1 <= 0
    
    # We can implement these as bounds or explicit constraints.
    # Bounds handle 0 <= x <= 1.
    # But x - r >= 0 is not a bound on x alone.
    # Let's use explicit inequality constraints g(z) >= 0.
    
    def boundary_constraints(z):
        con = []
        for i in range(n):
            x, y, r = z[3*i], z[3*i+1], z[3*i+2]
            # x - r >= 0
            con.append(x - r)
            # 1 - (x + r) >= 0
            con.append(1.0 - (x + r))
            # y - r >= 0
            con.append(y - r)
            # 1 - (y + r) >= 0
            con.append(1.0 - (y + r))
        return np.array(con)
        
    # 2. Non-overlap constraints:
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
    
    def non_overlap_constraints(z):
        con = []
        for i in range(n):
            xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                con.append(dist - (ri + rj))
        return np.array(con)

    # Combine constraints
    # scipy.optimize.minimize supports 'ineq' constraints in older versions, 
    # but NonlinearConstraint is preferred for SLSQP in newer scipy.
    # However, NonlinearConstraint expects a function returning an array and lb, ub.
    # g(z) >= 0 is equivalent to lb=0, ub=np.inf.
    
    # We can define a single function for all constraints.
    def all_constraints_func(z):
        c1 = boundary_constraints(z)
        c2 = non_overlap_constraints(z)
        return np.concatenate((c1, c2))
    
    n_constraints = 4 * n + n * (n - 1) // 2
    cons = NonlinearConstraint(all_constraints_func, 0, np.inf)
    
    # Run optimization
    # SLSQP is a good choice for this type of problem.
    # We might need to run it a few times or use a robust solver.
    # Given the non-convexity, one run might not be enough, but with a good init, it's a shot.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun > -2.5: # Check if we got a reasonable result
            centers_opt = np.zeros((n, 2))
            radii_opt = np.zeros(n)
            for i in range(n):
                centers_opt[i] = [res.x[3*i], res.x[3*i+1]]
                radii_opt[i] = res.x[3*i+2]
            
            # Post-process: Ensure radii are non-negative and valid
            # The constraints should handle this, but numerical errors might occur.
            radii_opt = np.maximum(radii_opt, 0)
            
            sum_radii = np.sum(radii_opt)
            return centers_opt, radii_opt, sum_radii
            
        else:
            # Fallback if optimization failed
            return centers, radii, np.sum(radii)
            
    except Exception as e:
        # Return initial guess if optimization crashes
        return centers, radii, np.sum(radii)
