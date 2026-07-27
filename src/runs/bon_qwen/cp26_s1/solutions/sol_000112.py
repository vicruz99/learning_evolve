# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eb34cb51) state=1dd88cb5 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n):
    """Objective function: minimize negative sum of radii."""
    radii = x[2*n:]
    return -np.sum(radii)

def compute_constraints(x, n):
    """
    Compute all inequality constraints g(x) >= 0.
    Returns an array of constraint values.
    """
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    cons = []
    
    # Boundary constraints: 0 <= x_i - r_i and x_i + r_i <= 1
    for i in range(n):
        xi, yi = centers[i]
        ri = radii[i]
        cons.append(xi - ri)
        cons.append(1.0 - (xi + ri))
        cons.append(yi - ri)
        cons.append(1.0 - (yi + ri))
        
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - (radii[i] + radii[j]))
            
    return np.array(cons)

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal-like pattern
    # Rows: 6, 5, 6, 5, 4 circles
    centers = np.zeros((n, 2))
    idx = 0
    row_counts = [6, 5, 6, 5, 4]
    
    # Approximate spacing to fit in [0.1, 0.9] initially
    y_step = 0.18
    x_step = 0.16
    
    for r_idx, cnt in enumerate(row_counts):
        if idx >= n: 
            break
        # Center vertically
        y = 0.5 + (r_idx - 2) * y_step
        for c_idx in range(cnt):
            # Center horizontally with offset for odd rows
            x = 0.5 + (c_idx - cnt/2 + 0.5) * x_step
            if r_idx % 2 == 1:
                x += x_step / 2.0
            centers[idx] = [x, y]
            idx += 1
            
    # Clip to ensure strictly inside bounds initially
    centers = np.clip(centers, 0.05, 0.95)
    
    # Initial radii: slightly smaller than target average to allow expansion
    radii = np.full(n, 0.09)
    
    # Flatten for optimizer
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Constraints object
    class PackingConstraints:
        def __init__(self, n):
            self.n = n
        def __call__(self, x):
            return compute_constraints(x, self.n)
            
    cons = {'type': 'ineq', 'fun': PackingConstraints(n)}
    
    # 2. Optimization
    # SLSQP is robust for this type of problem
    res = minimize(
        compute_objective, 
        x0, 
        args=(n,), 
        bounds=bounds, 
        constraints=cons, 
        method='SLSQP', 
        options={
            'maxiter': 2000, 
            'ftol': 1e-12, 
            'disp': False
        }
    )
    
    # Extract results
    if res.success or res.nit > 0:
        final_centers = res.x[:2*n].reshape(n, 2)
        final_radii = res.x[2*n:]
    else:
        # Fallback to initial if optimization fails completely
        final_centers = centers
        final_radii = radii
        
    # 3. Post-processing for numerical stability
    # Ensure strict feasibility within tolerance
    # Shrink radii slightly if any constraint is barely violated
    min_r = 1.0
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        # Boundary
        margin = min(x - r, 1 - (x + r), y - r, 1 - (y + r))
        if margin < 0:
            final_radii[i] += margin # Adjust radius to fit
            # Ensure non-negative
            final_radii[i] = max(0.0, final_radii[i])
            
    for i in range(n):
        for j in range(i+1, n):
            dx = final_centers[i,0] - final_centers[j,0]
            dy = final_centers[i,1] - final_centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = final_radii[i] + final_radii[j]
            if sum_r > dist:
                # Reduce both radii equally to resolve overlap
                delta = (sum_r - dist) / 2.0 + 1e-7
                final_radii[i] = max(0.0, final_radii[i] - delta)
                final_radii[j] = max(0.0, final_radii[j] - delta)

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
