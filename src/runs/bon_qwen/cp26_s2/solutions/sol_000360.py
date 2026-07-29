# sol_000360 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=fb98c167 sum of radii=2.166667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for number of circles
N_CIRCLES = 26

def objective(vars):
    """Objective function to minimize: -r (to maximize radius)"""
    return -vars[-1]

def constraints(vars):
    """Constraint functions: boundaries and non-overlap"""
    r = vars[-1]
    centers = vars[:-1].reshape(N_CIRCLES, 2)
    
    c_list = []
    # Boundary constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
    c_list.extend(centers[:, 0] - r)
    c_list.extend(centers[:, 1] - r)
    c_list.extend(1.0 - centers[:, 0] - r)
    c_list.extend(1.0 - centers[:, 1] - r)
    
    # Non-overlap constraints: dist(i, j) >= 2r
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            c_list.append(dist - 2 * r)
            
    return np.array(c_list)

def generate_hex_grid(seed=0):
    """Generates an initial hexagonal packing configuration"""
    np.random.seed(seed)
    r_init = 0.08
    pts = []
    row = 0
    while len(pts) < N_CIRCLES:
        offset = (row % 2) * r_init
        y = r_init + row * np.sqrt(3) * r_init
        x = r_init + offset
        while x <= 1.0 - r_init and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2 * r_init
        row += 1
    return np.array(pts[:N_CIRCLES])

def run_packing():
    # Setup bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum_r = 0.0
    best_centers = None
    best_radii = None
    
    # Try multiple initial configurations to avoid local minima
    for seed in range(5):
        init_centers = generate_hex_grid(seed)
        # Variable vector: [x0, y0, ..., x25, y25, r]
        x0 = np.concatenate([init_centers.flatten(), [0.08]])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            if res.success and -res.fun > best_sum_r:
                best_sum_r = -res.fun * N_CIRCLES
                best_centers = res.x[:-1].reshape(N_CIRCLES, 2)
                best_radii = np.full(N_CIRCLES, res.x[-1])
        except Exception:
            continue
            
    # Fallback to a valid configuration if optimization fails
    if best_centers is None:
        best_centers = generate_hex_grid(0)
        best_radii = np.full(N_CIRCLES, 0.08)
        best_sum_r = 0.08 * N_CIRCLES
        
    return best_centers, best_radii, best_sum_r
