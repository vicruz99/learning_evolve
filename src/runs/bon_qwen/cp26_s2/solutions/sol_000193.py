# sol_000193 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fb76805b) state=0c639e18 sum of radii=2.619449 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    best_result = None
    best_obj = -np.inf

    # Helper to create a hexagonal grid initial configuration
    def create_hex_grid(n):
        # Approximate radius for grid generation
        r_guess = 0.1
        # Hexagonal spacing
        dx = 2 * r_guess
        dy = np.sqrt(3) * r_guess
        
        centers = []
        # Fill rows
        row_idx = 0
        while len(centers) < n:
            y = r_guess + row_idx * dy
            # Check vertical boundary
            if y + r_guess > 1.0:
                # Try to squeeze rows closer or stop
                # If we can't fit more rows, break
                break
            
            # Determine number of circles in this row
            # Shift odd rows
            shift = 0.0
            if row_idx % 2 == 1:
                shift = dx / 2.0
            
            x = r_guess + shift
            while x + r_guess <= 1.0 and len(centers) < n:
                centers.append([x, y])
                x += dx
                # Safety break if no progress
                if x > 1.0 + 0.001: 
                    break
            
            row_idx += 1
        
        # If we didn't fill enough circles (unlikely with this logic for n=26), 
        # add some random ones in free space or just extend grid
        while len(centers) < n:
            # Place in a simple grid pattern if hex grid ran out
            # This fallback is rarely needed for 26
            idx = len(centers)
            gx = (idx % 5) + 1
            gy = (idx // 5) + 1
            centers.append([gx/10, gy/10])

        # Ensure we have exactly n centers
        centers = centers[:n]
        
        # Convert to numpy array
        C = np.array(centers)
        # Add small random noise to break symmetry
        noise = np.random.uniform(-0.01, 0.01, size=C.shape)
        C = C + noise
        # Clip to valid range for initial guess
        C = np.clip(C, 0.05, 0.95)
        
        return C, r_guess * np.ones(n)

    # Define objective function: minimize negative sum of radii
    def objective(vars):
        # vars is flat array [x1, y1, r1, x2, y2, r2, ...]
        # reshape to (n, 3)
        pts = vars.reshape((n_circles, 3))
        radii = pts[:, 2]
        return -np.sum(radii)

    # Define constraints
    # 1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # 2. Non-overlap: dist^2 >= (r1+r2)^2
    
    def get_constraints():
        constraints = []
        
        # Boundary constraints
        # For each circle i:
        # x_i - r_i >= 0  => r_i - x_i <= 0
        # 1 - x_i - r_i >= 0 => x_i + r_i - 1 <= 0
        # y_i - r_i >= 0 => r_i - y_i <= 0
        # 1 - y_i - r_i >= 0 => y_i + r_i - 1 <= 0
        
        # We will construct a function that returns a vector of constraint values <= 0
        # But SLSQP allows list of dict constraints. 
        # Vectorized constraint function is faster.
        
        def boundary_constraint(vars):
            pts = vars.reshape((n_circles, 3))
            xs = pts[:, 0]
            ys = pts[:, 1]
            rs = pts[:, 2]
            
            # Constraints: g(x) <= 0
            # r - x <= 0
            c1 = rs - xs
            # x + r - 1 <= 0
            c2 = xs + rs - 1.0
            # r - y <= 0
            c3 = rs - ys
            # y + r - 1 <= 0
            c4 = ys + rs - 1.0
            
            return np.concatenate([c1, c2, c3, c4])

        # Non-overlap constraints
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        # => (ri + rj)^2 - dist^2 <= 0
        
        def overlap_constraint(vars):
            pts = vars.reshape((n_circles, 3))
            rs = pts[:, 2]
            xs = pts[:, 0]
            ys = pts[:, 1]
            
            violations = []
            # Loop over pairs
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                    sum_r = rs[i] + rs[j]
                    # Constraint: sum_r^2 - dist_sq <= 0
                    violations.append(sum_r**2 - dist_sq)
            return np.array(violations)

        return {'type': 'ineq', 'fun': lambda v: -boundary_constraint(v)}, \
               {'type': 'ineq', 'fun': lambda v: -overlap_constraint(v)}

    # SLSQP requires constraints in a specific format. 
    # 'ineq' means fun(x) >= 0.
    # So we need fun(v) >= 0.
    # My derivations were for <= 0. So I negate them.
    
    # Let's bundle constraints properly
    constraints_list = []
    
    # Boundary: x >= r => x - r >= 0. 
    # Actually boundary is x >= r AND x <= 1-r.
    # So x - r >= 0 and 1 - r - x >= 0.
    def bound_constraints(v):
        pts = v.reshape((n_circles, 3))
        xs = pts[:, 0]
        ys = pts[:, 1]
        rs = pts[:, 2]
        # x - r >= 0
        c1 = xs - rs
        # 1 - r - x >= 0
        c2 = 1.0 - rs - xs
        # y - r >= 0
        c3 = ys - rs
        # 1 - r - y >= 0
        c4 = 1.0 - rs - ys
        return np.concatenate([c1, c2, c3, c4])
    
    constraints_list.append({'type': 'ineq', 'fun': bound_constraints})
    
    # Overlap: dist^2 - (r1+r2)^2 >= 0
    def overlap_constraints(v):
        pts = v.reshape((n_circles, 3))
        rs = pts[:, 2]
        xs = pts[:, 0]
        ys = pts[:, 1]
        vals = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                d2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                r_sum = rs[i] + rs[j]
                vals.append(d2 - r_sum**2)
        return np.array(vals)

    constraints_list.append({'type': 'ineq', 'fun': overlap_constraints})

    # Bounds for variables: x, y in [0, 1], r >= 0
    # Actually x, y bounds are handled by constraints, but we can set loose bounds
    lb = np.zeros(n_circles * 3)
    ub = np.ones(n_circles * 3)
    # r can be up to 0.5
    ub[2::3] = 0.5
    # x, y are in [0, 1]
    
    bounds = list(zip(lb, ub))

    # Run multiple times
    num_trials = 5
    
    for trial in range(num_trials):
        # Generate initial guess
        C, r_init = create_hex_grid(n_circles)
        
        # Flatten to vector
        x0 = np.zeros(n_circles * 3)
        for i in range(n_circles):
            x0[3*i] = C[i, 0]
            x0[3*i+1] = C[i, 1]
            x0[3*i+2] = r_init[i]
        
        # Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints_list, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_obj:
                    best_obj = current_sum
                    best_result = res
        except Exception as e:
            print(f"Optimization failed in trial {trial}: {e}")

    if best_result is None:
        # Fallback to a simple grid if optimization failed
        C, _ = create_hex_grid(n_circles)
        radii = 0.01 * np.ones(26)
        best_result_x = np.zeros(26*3)
        for i in range(26):
            best_result_x[3*i] = C[i,0]
            best_result_x[3*i+1] = C[i,1]
            best_result_x[3*i+2] = radii[i]
        return C, radii, np.sum(radii)

    # Extract result
    res_vars = best_result.x
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        centers[i, 0] = res_vars[3*i]
        centers[i, 1] = res_vars[3*i+1]
        radii[i] = res_vars[3*i+2]
    
    sum_radii = np.sum(radii)
    
    # Final validation check (silent)
    # The validate_packing function is provided, we can call it or trust the solver.
    # But to be safe, let's ensure non-negative radii explicitly
    radii = np.maximum(radii, 0)
    
    return centers, radii, sum_radii
