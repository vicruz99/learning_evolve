# sol_000221 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b6844e7) state=daa1f19f sum of radii=2.030555 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def _objective(vars_):
    """Objective function: minimize negative sum of radii."""
    r = vars_[2::3]
    return -np.sum(r)

def _constraints(vars_):
    """
    Returns array of constraint values. All must be >= 0 for feasibility.
    Constraints include boundary checks and pairwise non-overlap.
    """
    x = vars_[::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, etc.
    boundary_c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist(i,j) - r(i) - r(j) >= 0
    # Use broadcasting for O(N^2) calculation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    # Upper triangular mask to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    overlap_c = dist[mask] - r_sum[mask]
    
    return np.concatenate([boundary_c, overlap_c])

def _generate_initial_guess(n):
    """Generates a hexagonal-like initial placement for n circles."""
    r_init = 0.08  # Small enough to guarantee no initial overlap
    h = r_init * np.sqrt(3.0)
    w = 2.0 * r_init
    
    points = []
    rows = 6
    for i in range(rows):
        y = r_init + i * h
        # Stagger columns for hexagonal packing
        num_cols = 5 if i % 2 == 0 else 4
        for j in range(num_cols):
            x = r_init + j * w + (r_init if i % 2 == 1 else 0.0)
            points.append([x, y])
            
    # Ensure we have at least n points, trim if necessary
    if len(points) < n:
        # Fallback to grid if hex rows were too short
        step = 0.2
        for i in range(6):
            for j in range(6):
                if len(points) >= n:
                    break
                points.append([0.1 + j * step, 0.1 + i * step])
    
    pts = points[:n]
    
    # Construct initial variables vector [x1, y1, r1, x2, y2, r2, ...]
    init_vars = np.zeros(3 * n)
    for i, (x, y) in enumerate(pts):
        init_vars[3 * i] = x
        init_vars[3 * i + 1] = y
        init_vars[3 * i + 2] = r_init
        
    return init_vars

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # Constraint setup
    cons = {'type': 'ineq', 'fun': _constraints}
    
    # Initial guess
    x0 = _generate_initial_guess(n)
    
    # Run optimization
    # SLSQP is suitable for this constrained non-linear problem
    result = minimize(
        _objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
    )
    
    if not result.success:
        # Fallback if optimization fails, though it rarely does with this setup
        pass
        
    vars_opt = result.x
    centers = np.column_stack((vars_opt[::3], vars_opt[1::3]))
    radii = vars_opt[2::3]
    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)
