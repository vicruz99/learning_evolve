# sol_000379 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7a0a6c4a) state=de00a223 sum of radii=2.485950 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_overlap_constraints(vars_flat, n):
    """Returns array of (dist - r_i - r_j) for all i < j. Must be >= 0."""
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - radii[i] - radii[j])
    return np.array(cons)

def compute_boundary_constraints(vars_flat, n):
    """Returns array of boundary distances. Must be >= 0."""
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    cons = []
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        cons.extend([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
    return np.array(cons)

def objective_function(vars_flat, n):
    """Returns negative sum of radii for minimization."""
    radii = vars_flat[2*n:]
    return -np.sum(radii)

def run_packing():
    n = 26
    
    # 1. Initialize with perturbed hexagonal grid
    np.random.seed(42)
    centers = np.zeros((n, 2))
    radii_init = 0.04
    radii = np.full(n, radii_init)
    
    rows = 5
    cols = 6
    idx = 0
    x_spacing = 0.18
    y_spacing = 0.16
    
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                x = 0.12 + c * x_spacing + (x_spacing / 2 if r % 2 == 1 else 0.0)
                y = 0.12 + r * y_spacing
                centers[idx] = [x, y]
                idx += 1
                
    # Add small perturbation to break symmetry
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Define bounds
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': compute_overlap_constraints, 'args': (n,)},
        {'type': 'ineq', 'fun': compute_boundary_constraints, 'args': (n,)}
    ]
    
    # 2. Run constrained optimization
    res = minimize(objective_function, x0, args=(n,), method='SLSQP', 
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    # Extract results
    best_centers = res.x[:2*n].reshape(n, 2)
    best_radii = res.x[2*n:]
    
    # 3. Post-processing: Ensure strict validity within 1e-12 tolerance
    # Adjust radii to satisfy non-overlap exactly
    for _ in range(200):
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                req = best_radii[i] + best_radii[j] - dist
                if req > 1e-12:
                    # Reduce radii proportionally
                    scale_i = best_radii[i] / (best_radii[i] + best_radii[j])
                    scale_j = 1.0 - scale_i
                    best_radii[i] -= req * scale_i
                    best_radii[j] -= req * scale_j
                    max_viol += req
                    
        # Boundary adjustments
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if x - r < -1e-12: best_radii[i] = x
            if x + r > 1.0 + 1e-12: best_radii[i] = 1.0 - x
            if y - r < -1e-12: best_radii[i] = y
            if y + r > 1.0 + 1e-12: best_radii[i] = 1.0 - y
            
        if max_viol < 1e-13:
            break
            
    # Ensure non-negative radii
    best_radii = np.maximum(best_radii, 1e-9)
    
    sum_radii = np.sum(best_radii)
    return best_centers, best_radii, sum_radii
