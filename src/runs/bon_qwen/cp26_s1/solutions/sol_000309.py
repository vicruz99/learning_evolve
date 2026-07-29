# sol_000309 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39d28b7b) state=28f3a654 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def objective(params, n):
    # The objective is to maximize the sum of radii.
    # We minimize the negative sum.
    radii = params[2*n:]
    return -np.sum(radii)

def pair_constraints(p, pair_indices):
    # Computes the squared distance minus the squared sum of radii for all pairs.
    # Constraint: dist^2 >= (r_i + r_j)^2
    # Layout: [x0, y0, x1, y1, ..., xn-1, yn-1, r0, r1, ..., rn-1]
    # x_i is at 2*i, y_i is at 2*i + 1, r_i is at 2*n + i
    m = len(pair_indices)
    res = np.empty(m)
    k = 0
    for (xi, yi, ri, xj, yj, rj) in pair_indices:
        dx = p[xi] - p[xj]
        dy = p[yi] - p[yj]
        r_sum = p[ri] + p[rj]
        res[k] = dx*dx + dy*dy - r_sum*r_sum
        k += 1
    return res

def boundary_constraints(p, n):
    # Computes boundary constraints for all circles.
    # Constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    res = np.empty(4 * n)
    for i in range(n):
        xi = 2*i
        yi = 2*i + 1
        ri = 2*n + i
        
        r = p[ri]
        x = p[xi]
        y = p[yi]
        
        idx = 4 * i
        res[idx] = x - r
        res[idx+1] = 1.0 - x - r
        res[idx+2] = y - r
        res[idx+3] = 1.0 - y - r
    return res

def get_initial_configuration(n):
    # Generates a hexagonal lattice initialization for n circles
    centers = []
    radii = []
    r_est = 0.1
    y = r_est
    row = 0
    while len(centers) < n:
        x = r_est if row % 2 == 0 else 2 * r_est
        while x <= 1 - r_est and len(centers) < n:
            centers.append([x, y])
            radii.append(r_est)
            x += 2 * r_est
        y += r_est * np.sqrt(3)
        row += 1
    return np.array(centers[:n]), np.array(radii[:n])

def run_packing():
    n = 26
    
    best_params = None
    best_val = -np.inf
    
    # Precompute indices for pairwise constraints to speed up evaluation
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            # Indices in the flattened parameter vector
            pair_indices.append((2*i, 2*i+1, 2*n+i, 2*j, 2*j+1, 2*n+j))
            
    # Define constraints
    constraints = []
    constraints.append({'type': 'ineq', 'fun': pair_constraints, 'args': (pair_indices,)})
    constraints.append({'type': 'ineq', 'fun': boundary_constraints, 'args': (n,)})
    
    # Bounds for variables: coordinates in [0, 1], radii in [0, 1]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)] * n

    # Run optimization with multiple random restarts to avoid local minima
    for seed in range(10):
        np.random.seed(seed)
        c_init, r_init = get_initial_configuration(n)
        
        # Add small perturbation to initialization
        c_init += np.random.normal(0, 0.01, c_init.shape)
        c_init = np.clip(c_init, 0.05, 0.95)
        
        # Flatten centers and append radii to form the parameter vector
        x0 = np.concatenate([c_init.flatten(), r_init])
        
        try:
            res = opt.minimize(
                objective,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            # We minimized -sum(radii), so best_val stores max sum
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is not None:
        centers = best_params[:2*n].reshape(n, 2)
        radii = best_params[2*n:]
        
        # Post-processing correction to ensure strict validity (handling numerical errors)
        for _ in range(50):
            changed = False
            # Check boundary constraints
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                max_r = min(x, 1-x, y, 1-y)
                if r > max_r + 1e-10:
                    radii[i] = max_r
                    changed = True
            
            # Check pairwise overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < radii[i] + radii[j] - 1e-10:
                        overlap = (radii[i] + radii[j]) - d
                        # Reduce radii evenly to resolve overlap
                        radii[i] -= overlap / 2
                        radii[j] -= overlap / 2
                        changed = True
            if not changed:
                break
                
        return centers, radii, np.sum(radii)
    
    # Fallback to initial configuration if optimization fails
    centers, radii = get_initial_configuration(n)
    return centers, radii, np.sum(radii)
