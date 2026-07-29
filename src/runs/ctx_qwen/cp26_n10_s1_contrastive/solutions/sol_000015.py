# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 002dac80) state=cc21d5f7 sum of radii=2.591545 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """Compute all inequality constraints: boundary and separation."""
    N = 26
    C = x.reshape(N, 3)
    x_c = C[:, 0]
    y_c = C[:, 1]
    r = C[:, 2]
    
    c_list = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_list.append(x_c - r)
    c_list.append(1.0 - x_c - r)
    c_list.append(y_c - r)
    c_list.append(1.0 - y_c - r)
    
    # Separation constraints: dist_sq >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x_c[i_idx] - x_c[j_idx]
    dy = y_c[i_idx] - y_c[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c_list.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c_list)

def run_packing():
    n = 26
    
    # 1. Generate initial hexagonal grid layout
    rows_counts = [5, 6, 5, 6, 4]
    pts = []
    for row_idx, count in enumerate(rows_counts):
        y = row_idx * 1.4
        shift = (row_idx % 2) * 0.7
        for col in range(count):
            x = shift + col * 1.4
            pts.append([x, y])
    pts = np.array(pts)
    
    # Normalize to [0.1, 0.9] to leave room for expansion
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    pts = (pts - mn) / (mx - mn) * 0.8 + 0.1
    
    # 2. Compute safe initial radii
    dmin = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
            if d < dmin:
                dmin = d
        # Distance to boundaries
        dmin = min(dmin, pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        
    # Strictly feasible initial radii
    r_init = np.full(n, dmin / 6.0)
    
    # Flatten initial state: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = pts[i, 0]
        x0[3 * i + 1] = pts[i, 1]
        x0[3 * i + 2] = r_init[i]
        
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # 3. Optimize
    res = minimize(
        compute_objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': compute_constraints},
        options={'maxiter': 1500, 'ftol': 1e-11}
    )
    
    # 4. Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = res.x[3 * i], res.x[3 * i + 1]
        radii[i] = res.x[3 * i + 2]
        
    return centers, radii, np.sum(radii)
