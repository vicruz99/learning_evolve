# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d5d6e849) state=5455684e sum of radii=2.612115 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Optimization parameters
    num_restarts = 5
    max_iter = 1000
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    def get_initial_guess(method='random'):
        if method == 'random':
            # Place centers in a safe region [0.1, 0.9] to avoid immediate boundary conflicts
            # Start with small radii
            centers = np.random.uniform(0.1, 0.9, size=(n, 2))
            radii = np.full(n, 0.02)
        elif method == 'hex_grid':
            # Generate a hexagonal lattice
            # Estimate spacing for 26 circles: area ~ 1/26 -> r ~ 0.11 -> diameter ~ 0.22
            # Use spacing ~ 0.2
            pts = []
            dx = 0.2
            dy = dx * np.sqrt(3) / 2
            
            # Generate a grid slightly larger than the unit square
            for r in range(-5, 10):
                for c in range(-5, 10):
                    x = c * dx + (r % 2) * (dx / 2)
                    y = r * dy
                    # Check if inside [0, 1] x [0, 1] with some margin
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        pts.append([x, y])
            
            pts = np.array(pts)
            
            # If we have fewer points than needed, fall back or pad (unlikely with this grid)
            if len(pts) < n:
                centers = np.random.uniform(0.1, 0.9, size=(n, 2))
                radii = np.full(n, 0.02)
                return centers, radii
            
            # Select n points closest to the center of the square (0.5, 0.5)
            # This creates a compact cluster, which might be a good starting density
            center = np.array([0.5, 0.5])
            dists = np.linalg.norm(pts - center, axis=1)
            indices = np.argsort(dists)[:n]
            centers = pts[indices]
            
            # Estimate a reasonable initial radius based on nearest neighbors
            # Or just set small
            radii = np.full(n, 0.05)
            
        return centers, radii

    def objective(vars):
        # vars is flattened: [x1, y1, r1, x2, y2, r2, ...]
        # Extract radii
        r = vars[2::3]
        return -np.sum(r)

    def constraints(vars):
        # Reshape variables
        cs = np.array(vars).reshape(n, 3)
        x = cs[:, 0]
        y = cs[:, 1]
        r = cs[:, 2]
        
        c_list = []
        
        # 1. Boundary Constraints
        # Circle must be inside [0, 1] x [0, 1]
        # x - r >= 0  =>  r - x <= 0 (but we use >= 0 format for scipy ineq? No, standard is g(x) >= 0)
        # scipy 'ineq' constraint requires fun(x) >= 0.
        
        # x >= r  => x - r >= 0
        c_list.extend(x - r)
        # 1 - x >= r => 1 - x - r >= 0
        c_list.extend(1 - x - r)
        # y >= r  => y - r >= 0
        c_list.extend(y - r)
        # 1 - y >= r => 1 - y - r >= 0
        c_list.extend(1 - y - r)
        
        # 2. Non-overlap Constraints
        # dist(i, j) >= r_i + r_j  =>  dist - r_i - r_j >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.hypot(dx, dy)
                c_list.append(dist - r[i] - r[j])
        
        return np.array(c_list)

    bounds = []
    for i in range(n):
        # x in [0, 1], y in [0, 1], r in [0, 0.5] (theoretical max)
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    cons = {'type': 'ineq', 'fun': constraints}
    
    methods = ['random', 'hex_grid', 'random', 'random', 'random']
    
    for idx in range(num_restarts):
        method_type = methods[idx % len(methods)]
        centers_init, radii_init = get_initial_guess(method_type)
        
        # Flatten to 1D vector for scipy
        x0 = np.zeros(3 * n)
        x0[0::3] = centers_init[:, 0]
        x0[1::3] = centers_init[:, 1]
        x0[2::3] = radii_init
        
        try:
            res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP', 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
            
            # Check if optimization was successful or at least found a better feasible point
            # Note: res.success might be False even if a good point is found
            current_obj = res.fun
            current_sum = -current_obj
            
            # Validate the result manually to be safe
            if not np.isnan(current_obj):
                cs = res.x.reshape(n, 3)
                x = cs[:, 0]
                y = cs[:, 1]
                r = cs[:, 2]
                
                is_valid = True
                
                # Check bounds
                if np.any(x < r - 1e-10) or np.any(x + r > 1 + 1e-10) or \
                   np.any(y < r - 1e-10) or np.any(y + r > 1 + 1e-10):
                    is_valid = False
                
                # Check overlaps
                if is_valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = np.hypot(x[i] - x[j], y[i] - y[j])
                            if dist < r[i] + r[j] - 1e-10:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                
                if is_valid and current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = np.array([x, y]).T
                    best_radii = np.array(r)
        
        except Exception as e:
            # In case of error, continue to next restart
            pass

    # Fallback if no valid solution found (should not happen with random init)
    if best_centers is None:
        centers = np.random.uniform(0.2, 0.8, (n, 2))
        radii = np.full(n, 0.01)
        return centers, radii, np.sum(radii)

    return best_centers, best_radii, best_sum_radii
