# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 80fa60f2) state=14d1ad1d sum of radii=2.592939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(v, n):
    """Objective function: Minimize negative sum of radii."""
    # Radii are at indices 2, 5, 8, ... (3*i + 2)
    r = v[2::3]
    return -np.sum(r)

def constraint_function(v, n):
    """Inequality constraints: >= 0."""
    xs = v[0::3]
    ys = v[1::3]
    rs = v[2::3]
    
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    constraints.extend(xs - rs)
    constraints.extend(1.0 - xs - rs)
    constraints.extend(ys - rs)
    constraints.extend(1.0 - ys - rs)
    
    # Non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        xi, yi, ri = xs[i], ys[i], rs[i]
        for j in range(i + 1, n):
            xj, yj, rj = xs[j], ys[j], rs[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            r_sum = ri + rj
            constraints.append(dist_sq - r_sum*r_sum)
            
    return np.array(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initialize centers on a grid and add one in a gap
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    centers.append([0.2, 0.2]) # 26th circle in a gap
    
    centers = np.array(centers)
    radii = np.full(n, 0.05) # Small initial radius to ensure feasibility
    
    # Flatten to vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.empty(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    
    # Bounds for variables
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Define constraints object
    cons = {'type': 'ineq', 'fun': lambda v: constraint_function(v, n)}
    
    # Run optimization
    res = minimize(objective_function, x0, args=(n,), method='SLSQP', 
                   bounds=bounds, constraints=cons, 
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
    
    if not res.success:
        # Fallback to initial if optimization fails
        best_v = x0
    else:
        best_v = res.x
        
    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i] = [best_v[3*i], best_v[3*i+1]]
        final_radii[i] = best_v[3*i+2]
        
    total_radius = np.sum(final_radii)
    
    return final_centers, final_radii, total_radius
