# sol_000322 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ef4a4e64) state=899ff68a sum of radii=0.780000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[2::3])

def constraints(vars, N):
    """Returns array of constraint values that must be >= 0"""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    c = []
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise distance constraints: dist(i,j) - r[i] - r[j] >= 0
    for i in range(N):
        xi, yi, ri = x[i], y[i], r[i]
        for j in range(i + 1, N):
            dist = np.hypot(xi - x[j], yi - y[j])
            c.append(dist - ri - r[j])
            
    return np.concatenate(c)

def get_bounds(N):
    """Returns variable bounds for optimizer"""
    bounds = []
    for i in range(3 * N):
        if i % 3 == 2:
            # Radius bounds
            bounds.append((1e-5, 0.5))
        else:
            # Coordinate bounds
            bounds.append((0.0, 1.0))
    return bounds

def run_packing():
    """
    Optimizes circle packing to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    N = 26
    bounds = get_bounds(N)
    cons = {'type': 'ineq', 'fun': constraints, 'args': (N,)}

    best_vars = None
    best_sum = 0.0

    # --- Initialization Strategies ---
    init_configs = []

    # 1. Hexagonal grid pattern
    pts = []
    y_pos = 0.1
    while len(pts) < N:
        x_pos = 0.1
        while x_pos < 0.9 and len(pts) < N:
            pts.append([x_pos, y_pos])
            x_pos += np.sqrt(3) * 0.08
        y_pos += 0.08 * 1.5
    init_centers = np.array(pts[:N])
    init_radii = np.full(N, 0.03)
    v0_hex = np.zeros(3 * N)
    v0_hex[0::3] = init_centers[:, 0]
    v0_hex[1::3] = init_centers[:, 1]
    v0_hex[2::3] = init_radii
    init_configs.append(v0_hex)

    # 2. Random distribution
    np.random.seed(42)
    rand_centers = np.random.uniform(0.1, 0.9, (N, 2))
    rand_radii = np.full(N, 0.03)
    v0_rand = np.zeros(3 * N)
    v0_rand[0::3] = rand_centers[:, 0]
    v0_rand[1::3] = rand_centers[:, 1]
    v0_rand[2::3] = rand_radii
    init_configs.append(v0_rand)

    # --- Optimization Loop ---
    for vp in init_configs:
        for _ in range(5):
            # Perturb current configuration
            vp_trial = vp + np.random.normal(0, 0.005, 3 * N)
            vp_trial[0::3] = np.clip(vp_trial[0::3], 0.01, 0.99)
            vp_trial[1::3] = np.clip(vp_trial[1::3], 0.01, 0.99)
            vp_trial[2::3] = np.clip(vp_trial[2::3], 1e-4, 0.49)

            try:
                res = minimize(objective, vp_trial, method='SLSQP', bounds=bounds, 
                               constraints=cons, options={'maxiter': 1200, 'ftol': 1e-12, 'disp': False})
                curr_sum = -res.fun
                
                # Validate candidate solution before accepting
                c_test = res.x[0::3]
                y_test = res.x[1::3]
                r_test = res.x[2::3]
                
                boundary_valid = (np.all(r_test >= 1e-7) and 
                                  np.all(c_test >= r_test) and np.all(c_test <= 1 - r_test) and 
                                  np.all(y_test >= r_test) and np.all(y_test <= 1 - r_test))
                
                if curr_sum > best_sum and boundary_valid:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization failed completely
    if best_vars is None:
        best_vars = init_configs[0]

    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3].copy()

    # --- Repair Step ---
    # Ensures strict validity against the validation function's tolerance
    for _ in range(100):
        valid = True
        # Enforce boundary constraints
        for i in range(N):
            radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], 
                           centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] < 1e-8:
                valid = False

        # Resolve overlaps by proportional shrinking
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j]:
                    factor = d / (radii[i] + radii[j])
                    radii[i] *= factor * 0.9999
                    radii[j] *= factor * 0.9999
                    valid = False
        if valid:
            break

    return centers, radii, np.sum(radii)
