# sol_000227 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f043a2e3) state=95a104b7 sum of radii=2.608167 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization on a hexagonal initialization.
    """
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Hexagonal grid pattern
    # We generate more points than needed and select the first n
    pts = []
    for i in range(7):
        for j in range(6):
            # Hexagonal spacing
            x = 0.06 + i * 0.145 + (j % 2) * 0.0725
            y = 0.06 + j * 0.125
            if x <= 0.94 and y <= 0.94:
                pts.append([x, y])
    
    pts = pts[:n]
    centers = np.array(pts)
    # Small perturbation to break symmetry and help escape local minima
    centers = centers + np.random.randn(n, 2) * 0.002
    centers = np.clip(centers, 0.02, 0.98)
    
    # Initial radii: slightly conservative to ensure feasibility
    radii = np.ones(n) * 0.075
    
    # Flatten variables: [x1, y1, ..., xn, yn, r1, ..., rn]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(vars):
        return -np.sum(vars[2*n:])
        
    # Constraint: Non-overlap between all pairs
    def constraint_overlap(vars):
        c = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Sum of radii matrix
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Extract upper triangle (i < j)
        idx = np.triu_indices(n, k=1)
        return dists[idx] - r_sum[idx]
        
    # Constraint: Circles inside unit square
    def constraint_boundary(vars):
        c = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        return np.concatenate([
            c[:, 0] - r,          # x >= r
            1.0 - c[:, 0] - r,    # 1-x >= r
            c[:, 1] - r,          # y >= r
            1.0 - c[:, 1] - r     # 1-y >= r
        ])
        
    constraints = [
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_boundary}
    ]
    
    # Run optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-11, 'disp': False}
    )
    
    # Extract results
    centers_opt = res.x[:2*n].reshape(n, 2)
    radii_opt = res.x[2*n:]
    
    # Ensure non-negative radii and handle tiny numerical violations
    radii_opt = np.clip(radii_opt, 0.0, None)
    
    # Verify and adjust boundaries strictly
    for i in range(n):
        r = radii_opt[i]
        x, y = centers_opt[i]
        # Project to feasible region if needed
        centers_opt[i, 0] = np.clip(x, r, 1.0 - r)
        centers_opt[i, 1] = np.clip(y, r, 1.0 - r)
        
    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii
