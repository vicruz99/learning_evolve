# sol_000331 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 488bfafc) state=a62bfacb sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # Helper function to create constraints
    def get_constraints_and_bounds():
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        # We stack bounds: [x1, y1, r1, x2, y2, r2, ...]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((1e-6, 0.5)) # r (lower bound slightly > 0 to avoid numerical issues)
            
        constraints = []
        
        # Boundary constraints function
        # Returns array of size 4*n where each element must be >= 0
        def boundary_constraints(vars_flat):
            vals = np.zeros(4 * n)
            for i in range(n):
                idx = 3 * i
                x = vars_flat[idx]
                y = vars_flat[idx + 1]
                r = vars_flat[idx + 2]
                
                # x >= r  => x - r >= 0
                vals[4 * i] = x - r
                # x <= 1 - r => 1 - x - r >= 0
                vals[4 * i + 1] = 1.0 - x - r
                # y >= r => y - r >= 0
                vals[4 * i + 2] = y - r
                # y <= 1 - r => 1 - y - r >= 0
                vals[4 * i + 3] = 1.0 - y - r
            return vals

        # Separation constraints function
        # Returns array of size n*(n-1)/2 where each element must be >= 0
        # Constraint: dist^2 >= (r_i + r_j)^2
        def separation_constraints(vars_flat):
            # Extract arrays for vectorized operations where possible
            # Though loops are simple enough for n=26
            x = vars_flat[0:3*n:3] 
            y = vars_flat[1:3*n:3] 
            r = vars_flat[2:3*n:3] 
            
            num_pairs = n * (n - 1) // 2
            vals = np.zeros(num_pairs)
            
            pair_idx = 0
            for i in range(n):
                xi, yi, ri = x[i], y[i], r[i]
                # Vectorize inner loop slightly?
                # dx = x[i+1:] - xi
                # dy = y[i+1:] - yi
                # dr = r[i+1:] + ri
                # dist_sq = dx**2 + dy**2
                # min_dist_sq = dr**2
                # vals[pair_idx : pair_idx + (n-1-i)] = dist_sq - min_dist_sq
                # pair_idx += (n-1-i)
                
                # Manual loop for clarity and safety
                for j in range(i + 1, n):
                    dist_sq = (xi - x[j])**2 + (yi - y[j])**2
                    min_dist_sq = (ri + r[j])**2
                    vals[pair_idx] = dist_sq - min_dist_sq
                    pair_idx += 1
            return vals

        constraints.append({
            'type': 'ineq',
            'fun': boundary_constraints
        })
        
        constraints.append({
            'type': 'ineq',
            'fun': separation_constraints
        })
        
        return bounds, constraints

    def generate_initial_guess():
        # Hexagonal packing initialization
        # Target sum ~ 2.636 => avg r ~ 0.101.
        # Start with slightly smaller r to ensure valid initial guess.
        r_est = 0.095
        centers = []
        
        row_y = r_est
        row_idx = 0
        
        while len(centers) < n:
            # Horizontal spacing in hexagonal packing is 2r
            dx = 2 * r_est
            
            if row_idx % 2 == 0:
                # Even row: centers at r_est, 3r_est, ...
                start_x = r_est
                # Check how many fit: x_k = start_x + k*dx <= 1 - r_est
                # k*dx <= 1 - 2*r_est
                if 1 - 2 * r_est < 0:
                    count_row = 0
                else:
                    count_row = int((1 - 2 * r_est) / dx) + 1
            else:
                # Odd row: shifted by r_est relative to even row centers?
                # In hex packing, row 1 is shifted by dx/2 = r_est relative to row 0.
                # Row 0 centers: r, 3r, 5r...
                # Row 1 centers: 2r, 4r, 6r...
                # So start_x = 2*r_est
                start_x = 2 * r_est
                # Check fit: x_k <= 1 - r_est
                # k*dx <= 1 - r_est - 2*r_est = 1 - 3*r_est
                if 1 - 3 * r_est < 0:
                    count_row = 0
                else:
                    count_row = int((1 - 3 * r_est) / dx) + 1
            
            for k in range(count_row):
                if len(centers) < n:
                    x = start_x + k * dx
                    centers.append([x, row_y])
                else:
                    break
            
            # Vertical spacing is sqrt(3)/2 * diameter = sqrt(3) * r
            row_y += np.sqrt(3) * r_est
            row_idx += 1
            
        centers = np.array(centers[:n])
        radii = np.full(n, r_est)
        return centers, radii

    # Generate initial guess
    centers, radii = generate_initial_guess()
    
    # Flatten to vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Get bounds and constraints
    bounds, constraints = get_constraints_and_bounds()
    
    # Objective: maximize sum(r) => minimize -sum(r)
    def objective(vars_flat):
        radii = vars_flat[2::3]
        return -np.sum(radii)
        
    # Run optimizer
    # SLSQP is suitable for this type of problem
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False})
        vars_opt = res.x
    except Exception:
        # Fallback
        vars_opt = x0

    # Extract results
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        centers_opt[i, 0] = vars_opt[3*i]
        centers_opt[i, 1] = vars_opt[3*i+1]
        radii_opt[i] = vars_opt[3*i+2]
        
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
