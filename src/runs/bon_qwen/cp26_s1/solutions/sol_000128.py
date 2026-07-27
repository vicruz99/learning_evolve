# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92361807) state=70d41efa sum of radii=2.404698 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def objective_func(v):
    """Minimize negative sum of radii"""
    return -v[2::3].sum()

def boundary_constraints_func(v):
    """Ensure circles are inside the unit square"""
    n = len(v) // 3
    xs = v[0::3]
    ys = v[1::3]
    rs = v[2::3]
    return np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])

def overlap_constraints_func(v):
    """Ensure circles do not overlap"""
    n = len(v) // 3
    xs = v[0::3]
    ys = v[1::3]
    rs = v[2::3]
    
    # Vectorized pairwise distances and min allowed distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx**2 + dy**2
    
    r_sum = rs[:, None] + rs[None, :]
    min_d2 = r_sum**2
    
    # Extract upper triangle to avoid duplicates and self-pairs
    idx = np.triu_indices(n, k=1)
    return d2[idx] - min_d2[idx]

def run_packing():
    n = 26
    
    # Initial hexagonal packing configuration
    # Row lengths sum to 26: 5+5+5+5+4+2 = 26
    rows = [5, 5, 5, 5, 4, 2]
    centers = []
    radii_list = []
    
    # Start with a strictly feasible radius to help optimizer converge
    r_init = 0.055
    y = r_init
    for i, n_c in enumerate(rows):
        # Center the row horizontally
        x_start = (1.0 - (n_c - 1) * 2 * r_init - 2 * r_init) / 2.0
        if i % 2 == 1:
            x_start += r_init  # Offset for hexagonal pattern
        for k in range(n_c):
            x = x_start + k * 2 * r_init
            centers.append([x, y])
            radii_list.append(r_init)
        y += r_init * np.sqrt(3)
        
    centers = np.array(centers)
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii_list
    
    # Variable bounds: 0 <= x,y <= 1, r > 0
    bounds = [(0.0, 1.0)] * 2*n + [(1e-9, 0.5)] * n
    
    # Define constraints using trust-constr compatible objects
    cons_boundary = NonlinearConstraint(boundary_constraints_func, np.zeros(4*n), np.inf)
    cons_overlap = NonlinearConstraint(overlap_constraints_func, np.zeros(n*(n-1)//2), np.inf)
    
    # Run optimization
    res = minimize(objective_func, x0, method='trust-constr', bounds=bounds, 
                   constraints=[cons_boundary, cons_overlap], 
                   options={'maxiter': 5000, 'gtol': 1e-12, 'verbose': 0})
                   
    x_opt = res.x
    opt_centers = x_opt.reshape(-1, 3)[:, :2]
    opt_radii = x_opt[2::3]
    
    # Safety clamp for radii
    opt_radii = np.maximum(opt_radii, 1e-9)
    
    return opt_centers, opt_radii, float(np.sum(opt_radii))
