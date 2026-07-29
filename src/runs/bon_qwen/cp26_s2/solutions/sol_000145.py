# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6c47fa47) state=f60b9e37 sum of radii=2.617196 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """
    Objective function: Minimize negative sum of radii (maximize sum of radii).
    v is the flattened vector [x1, y1, r1, x2, y2, r2, ...]
    """
    return -np.sum(v[2::3])

def inequality_constraints(v, n):
    """
    Returns an array of constraint values that must be >= 0.
    Includes boundary constraints and non-overlap constraints.
    """
    xs = v[0::3]
    ys = v[1::3]
    rs = v[2::3]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = [
        xs - rs,
        1 - xs - rs,
        ys - rs,
        1 - ys - rs
    ]
    
    # Non-overlap constraints: dist(i, j) - (r_i + r_j) >= 0 for all i < j
    # Vectorized computation for efficiency
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    r_sum = rs[:, None] + rs[None, :]
    
    # Mask for upper triangle (i < j)
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    c.append(dists[tri] - r_sum[tri])
    
    return np.concatenate(c)

def run_packing():
    n = 26
    
    # Initialization: Grid pattern with noise
    # A 5x5 grid gives 25 points. We add one extra point.
    # Linspace ensures points are well spread.
    x_vals = np.linspace(0.15, 0.85, 5)
    y_vals = np.linspace(0.15, 0.85, 5)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Add 26th point near the bottom edge center
    extra_point = np.array([[0.5, 0.05]])
    centers = np.vstack((centers, extra_point))
    
    # Add small random noise to break symmetry and avoid grid artifacts
    # Using fixed seed for reproducibility
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to ensure centers are within reasonable bounds initially
    centers = np.clip(centers, 0.05, 0.95)
    
    # Initial radii: Small value to ensure valid starting configuration
    radii = np.full(n, 0.05)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    bounds = tuple(bounds)
    
    # Optimization
    # Using SLSQP method for constrained nonlinear optimization
    res = minimize(
        objective, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints={'type': 'ineq', 'fun': inequality_constraints, 'args': (n,)},
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )
    
    # Extract results
    final_x = res.x
    centers_out = np.column_stack((final_x[0::3], final_x[1::3]))
    radii_out = final_x[2::3]
    
    # Ensure radii are non-negative (safety clamp)
    radii_out = np.maximum(radii_out, 0)
    
    # Clamp centers to [0, 1] to satisfy boundary strictly
    centers_out = np.clip(centers_out, 0, 1)
    
    sum_radii = np.sum(radii_out)
    
    return centers_out, radii_out, sum_radii
