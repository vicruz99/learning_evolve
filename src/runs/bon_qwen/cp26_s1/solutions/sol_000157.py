# sol_000157 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bb08abb) state=3dfe0df7 sum of radii=2.626362 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Helper to create constraints
    def get_constraints(n_circles):
        constraints = []
        
        # Boundary constraints for each circle
        # x >= r  => x - r >= 0
        # 1 - x - r >= 0
        # y >= r  => y - r >= 0
        # 1 - y - r >= 0
        
        for i in range(n_circles):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, ix=idx_x, ir=idx_r: v[ix] - v[ir],
                'jac': lambda v, i=i, ix=idx_x, ir=idx_r: np.array([
                    1.0 if j == ix else (-1.0 if j == ir else 0.0) for j in range(3 * n_circles)
                ])
            })
            
            # 1 - x - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, ix=idx_x, ir=idx_r: 1.0 - v[ix] - v[ir],
                'jac': lambda v, i=i, ix=idx_x, ir=idx_r: np.array([
                    -1.0 if j == ix or j == ir else 0.0 for j in range(3 * n_circles)
                ])
            })
            
            # y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, iy=idx_y, ir=idx_r: v[iy] - v[ir],
                'jac': lambda v, i=i, iy=idx_y, ir=idx_r: np.array([
                    1.0 if j == iy else (-1.0 if j == ir else 0.0) for j in range(3 * n_circles)
                ])
            })
            
            # 1 - y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, iy=idx_y, ir=idx_r: 1.0 - v[iy] - v[ir],
                'jac': lambda v, i=i, iy=idx_y, ir=idx_r: np.array([
                    -1.0 if j == iy or j == ir else 0.0 for j in range(3 * n_circles)
                ])
            })
        
        # Non-overlap constraints for all pairs
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                idx_xi, idx_yi, idx_ri = 3 * i, 3 * i + 1, 3 * i + 2
                idx_xj, idx_yj, idx_rj = 3 * j, 3 * j + 1, 3 * j + 2
                
                # Function value
                def fun(v, i=i, j=j, x_i=idx_xi, y_i=idx_yi, r_i=idx_ri, 
                                    x_j=idx_xj, y_j=idx_yj, r_j=idx_rj):
                    dx = v[x_i] - v[x_j]
                    dy = v[y_i] - v[y_j]
                    dr = v[r_i] + v[r_j]
                    return dx*dx + dy*dy - dr*dr
                
                # Jacobian
                def jac(v, i=i, j=j, x_i=idx_xi, y_i=idx_yi, r_i=idx_ri, 
                                    x_j=idx_xj, y_j=idx_yj, r_j=idx_rj):
                    grad = np.zeros(3 * n_circles)
                    dx = v[x_i] - v[x_j]
                    dy = v[y_i] - v[y_j]
                    dr = v[r_i] + v[r_j]
                    
                    # Gradient for i
                    grad[x_i] = 2.0 * dx
                    grad[y_i] = 2.0 * dy
                    grad[r_i] = -2.0 * dr
                    
                    # Gradient for j
                    grad[x_j] = -2.0 * dx
                    grad[y_j] = -2.0 * dy
                    grad[r_j] = -2.0 * dr
                    
                    return grad
                
                constraints.append({
                    'type': 'ineq',
                    'fun': fun,
                    'jac': jac
                })
        
        return constraints

    def objective(v, n_circles):
        # Minimize negative sum of radii
        r_sum = sum(v[3*i + 2] for i in range(n_circles))
        return -r_sum

    def obj_jac(v, n_circles):
        grad = np.zeros(3 * n_circles)
        for i in range(n_circles):
            grad[3 * i + 2] = -1.0
        return grad

    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    # Initialization
    # Start with a 5x5 grid (25 circles) + 1 in a hole
    # Grid positions
    centers_init = []
    radii_init = []
    
    grid_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    for y in grid_coords:
        for x in grid_coords:
            centers_init.append([x, y])
            radii_init.append(0.1)
    
    # Add 26th circle in a hole, e.g., at (0.2, 0.2)
    # Distance to nearest neighbors (0.1, 0.1) etc is sqrt(0.02) ~ 0.1414
    # Radius can be ~0.0414. Use 0.04 to be safe.
    centers_init.append([0.2, 0.2])
    radii_init.append(0.04)
    
    # Random restarts to avoid local minima
    best_v = None
    best_obj = float('inf')
    
    # Generate constraints once (structure is same)
    constraints = get_constraints(n)
    
    # Try a few random perturbations
    np.random.seed(42)
    for trial in range(5):
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i][0]
            x0[3*i+1] = centers_init[i][1]
            x0[3*i+2] = radii_init[i]
            
        # Add small noise
        noise = np.random.normal(0, 0.005, size=3*n)
        # Keep radii positive and small noise
        noise[2::3] *= 0.5 
        x0_noisy = x0 + noise
        
        # Clip to bounds
        for i in range(n):
            x0_noisy[3*i] = np.clip(x0_noisy[3*i], 0, 1)
            x0_noisy[3*i+1] = np.clip(x0_noisy[3*i+1], 0, 1)
            x0_noisy[3*i+2] = np.clip(x0_noisy[3*i+2], 0, 0.5)
            
        # Optimize
        try:
            res = minimize(
                objective, 
                x0_noisy, 
                args=(n,),
                method='SLSQP',
                jac=obj_jac,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x
        except Exception:
            continue

    # If optimization failed or didn't improve, use initial
    if best_v is None:
        # Fallback to initial valid config (flattened)
        best_v = np.zeros(3 * n)
        for i in range(n):
            best_v[3*i] = centers_init[i][0]
            best_v[3*i+1] = centers_init[i][1]
            best_v[3*i+2] = radii_init[i]

    # Extract results
    centers_res = np.zeros((n, 2))
    radii_res = np.zeros(n)
    
    for i in range(n):
        centers_res[i, 0] = best_v[3*i]
        centers_res[i, 1] = best_v[3*i+1]
        radii_res[i] = best_v[3*i+2]
    
    sum_radii = np.sum(radii_res)
    
    return centers_res, radii_res, sum_radii
