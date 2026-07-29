# sol_000201 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=64eedbc2 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars):
    """Objective: maximize sum of radii => minimize negative sum."""
    radii = vars[2::3]
    return -np.sum(radii)

def compute_constraints(vars):
    """Compute all inequality constraints g(vars) >= 0."""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    # x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = x - r
    c2 = 1.0 - x - r
    c3 = y - r
    c4 = 1.0 - y - r
    c5 = r  # radii non-negativity
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for all pairs
    X = x[:, np.newaxis] - x[np.newaxis, :]
    Y = y[:, np.newaxis] - y[np.newaxis, :]
    R_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    dist_sq = X**2 + Y**2
    overlap_vals = dist_sq - R_sum**2
    
    # Extract upper triangle (i < j) to avoid duplicates and self-checks
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c6 = overlap_vals[mask]
    
    return np.concatenate([c1, c2, c3, c4, c5, c6])

def get_initial_guess():
    """Generate a feasible initial configuration using a hexagonal lattice."""
    n = 26
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.10
    
    idx = 0
    rows = [6, 5, 6, 5, 4]
    spacing = 0.16
    y_base = 0.12
    
    for i, count in enumerate(rows):
        y = y_base + i * spacing * np.sqrt(3) / 2.0
        x_start = 0.12 + (0.0 if i % 2 == 0 else spacing / 2.0)
        for j in range(count):
            if idx < n:
                centers[idx, 0] = x_start + j * spacing
                centers[idx, 1] = y
                idx += 1
                
    # Flatten to [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    vars0 = np.empty(3 * n)
    vars0[0::3] = centers[:, 0]
    vars0[1::3] = centers[:, 1]
    vars0[2::3] = radii
    return vars0

def run_packing():
    n = 26
    vars0 = get_initial_guess()
    
    # Define bounds: x,y in [0,1], r in [0,0.5]
    bnds = []
    for i in range(3 * n):
        if i % 3 == 2:
            bnds.append((0.0, 0.5))
        else:
            bnds.append((0.0, 1.0))
            
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Run SLSQP optimizer
    res = minimize(compute_objective, vars0, method='SLSQP', 
                   bounds=bnds, constraints=cons, 
                   options={'maxiter': 3000, 'ftol': 1e-12})
                   
    # Extract results
    centers = np.zeros((n, 2))
    centers[:, 0] = res.x[0::3]
    centers[:, 1] = res.x[1::3]
    radii = res.x[2::3]
    
    # Ensure non-negative radii (handle potential numerical noise)
    radii = np.maximum(radii, 0.0)
    
    total_sum = np.sum(radii)
    return centers, radii, total_sum
