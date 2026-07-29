# sol_000237 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e58a758a) state=8945d528 sum of radii=2.608050 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(x, n):
    """
    Computes all inequality constraints for the circle packing problem.
    Returns an array of constraint values that must be >= 0.
    x is a flattened array: [x1, y1, ..., xn, yn, r1, ..., rn]
    """
    cons = []
    
    # Boundary constraints: 4 per circle
    for i in range(n):
        xi, yi, ri = 2*i, 2*i+1, 2*n+i
        cons.append(x[xi] - x[ri])          # x >= r
        cons.append(1.0 - x[xi] - x[ri])    # 1-x >= r
        cons.append(x[yi] - x[ri])          # y >= r
        cons.append(1.0 - x[yi] - x[ri])    # 1-y >= r
        
    # Overlap constraints: 1 per pair
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = 2*i, 2*i+1, 2*n+i
            xj, yj, rj = 2*j, 2*j+1, 2*n+j
            dx = x[xi] - x[xj]
            dy = x[yi] - x[yj]
            dist_sq = dx*dx + dy*dy
            sum_r = x[ri] + x[rj]
            cons.append(dist_sq - sum_r*sum_r)  # d^2 >= (r1+r2)^2
            
    return np.array(cons)

def objective_func(x, n):
    """
    Objective function to maximize sum of radii.
    Returns negative sum since scipy minimizes.
    """
    return -np.sum(x[2*n:])

def run_packing():
    n = 26
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(2024)
    
    # Generate multiple initial configurations to avoid local minima
    starts = []
    
    # 1. Hexagonal grid packing
    hex_pts = []
    s = 0.22  # approximate spacing
    for r in range(7):
        for c in range(7):
            if len(hex_pts) >= n: break
            x = c * s + (r % 2) * s/2 + 0.15
            y = r * s * np.sqrt(3)/2 + 0.15
            hex_pts.append([x, y])
        if len(hex_pts) >= n: break
    starts.append(np.array(hex_pts[:n]))
    
    # 2. Random uniform placement
    starts.append(rng.uniform(0.1, 0.9, size=(n, 2)))
    
    # 3. Perturbed hexagonal grid
    starts.append(np.clip(hex_pts[:n].copy() + rng.normal(0, 0.02, size=(n, 2)), 0.05, 0.95))
    
    # Bounds for variables: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    for init_c in starts:
        # Initial radii set small to ensure feasibility
        x0 = np.concatenate([init_c.flatten(), np.full(n, 0.04)])
        
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP',
                           bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n,)},
                           options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            
            if res.success:
                score = np.sum(res.x[2*n:])
                if score > best_score:
                    best_score = score
                    best_centers = res.x[:2*n].reshape(n, 2)
                    best_radii = res.x[2*n:]
        except Exception:
            continue
            
    # Fallback if optimization fails (should not happen with valid starts)
    if best_centers is None:
        best_centers = starts[0]
        best_radii = np.full(n, 0.04)
        best_score = np.sum(best_radii)
        
    # Ensure strict feasibility within numerical tolerance
    for i in range(n):
        best_centers[i, 0] = np.clip(best_centers[i, 0], best_radii[i], 1.0 - best_radii[i])
        best_centers[i, 1] = np.clip(best_centers[i, 1], best_radii[i], 1.0 - best_radii[i])
        best_radii[i] = max(best_radii[i], 0.0)
        
    return best_centers, best_radii, best_score
