# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=853d481a sum of radii=2.614839 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars_array, n):
    """Computes inequality constraints for boundary and non-overlap."""
    x = vars_array[0::3]
    y = vars_array[1::3]
    r = vars_array[2::3]
    
    c_list = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dists = np.sqrt(x_diff**2 + y_diff**2)
    
    r_sum = r[:, None] + r[None, :]
    
    rows, cols = np.triu_indices(n, k=1)
    c_list.append(dists[rows, cols] - r_sum[rows, cols])
    
    return np.concatenate(c_list)

def objective(vars_array):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_array[2::3])

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds: x, y in [0, 1], r in [small_positive, 0.5]
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    inits = []
    
    # 1. Hexagonal initialization
    r_init = 0.10
    pts = []
    y = r_init
    row = 0
    while len(pts) < n and y + r_init <= 1.0:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
        
    v = np.zeros(3*n)
    v[0::3] = [p[0] for p in pts]
    v[1::3] = [p[1] for p in pts]
    v[2::3] = [r_init] * n
    inits.append(v)
    
    # 2. Grid initialization
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    grid_pts.append([0.5, 0.5])
    v2 = np.zeros(3*n)
    v2[0::3] = [p[0] for p in grid_pts]
    v2[1::3] = [p[1] for p in grid_pts]
    v2[2::3] = [0.09] * n
    inits.append(v2)
    
    # 3. Perturbed Hex initialization
    np.random.seed(123)
    noise = np.random.uniform(-0.015, 0.015, size=(n, 2))
    p3 = np.array(pts[:n]) + noise
    p3 = np.clip(p3, 0.05, 0.95)
    v3 = np.zeros(3*n)
    v3[0::3] = p3[:, 0]
    v3[1::3] = p3[:, 1]
    v3[2::3] = [0.095] * n
    inits.append(v3)
    
    # 4. Random initializations
    for seed in range(5, 12):
        np.random.seed(seed)
        v4 = np.zeros(3*n)
        v4[0::3] = np.random.uniform(0.1, 0.9, n)
        v4[1::3] = np.random.uniform(0.1, 0.9, n)
        v4[2::3] = np.full(n, 0.08)
        inits.append(v4)
        
    for init in inits:
        try:
            res = minimize(objective, init, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            
            if res.success:
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                
                # Validation
                valid = True
                if np.any(r_opt < 1e-7): valid = False
                if np.any(x_opt - r_opt < -1e-9) or np.any(x_opt + r_opt > 1 + 1e-9): valid = False
                if np.any(y_opt - r_opt < -1e-9) or np.any(y_opt + r_opt > 1 + 1e-9): valid = False
                
                centers = np.column_stack((x_opt, y_opt))
                dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
                r_sums = r_opt[:, None] + r_opt[None, :]
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.any(dists[mask] < r_sums[mask] - 1e-9):
                    valid = False
                
                if valid:
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers
                        best_radii = r_opt
        except Exception:
            continue

    if best_centers is None:
        best_centers = np.array(pts[:n])
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
