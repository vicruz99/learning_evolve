# sol_000278 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=0352edc3 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_arr):
    # Minimize negative sum of radii to maximize sum of radii
    return -np.sum(vars_arr[2::3])

def boundary_constraints(vars_arr):
    n = 26
    c = np.zeros(4 * n)
    for i in range(n):
        idx = 3 * i
        x, y, r = vars_arr[idx], vars_arr[idx+1], vars_arr[idx+2]
        base = 4 * i
        c[base] = x - r
        c[base+1] = 1.0 - x - r
        c[base+2] = y - r
        c[base+3] = 1.0 - y - r
    return c

def pairwise_constraints(vars_arr):
    n = 26
    m = n * (n - 1) // 2
    c = np.zeros(m)
    k = 0
    for i in range(n):
        idx_i = 3 * i
        xi, yi, ri = vars_arr[idx_i], vars_arr[idx_i+1], vars_arr[idx_i+2]
        for j in range(i + 1, n):
            idx_j = 3 * j
            xj, yj, rj = vars_arr[idx_j], vars_arr[idx_j+1], vars_arr[idx_j+2]
            c[k] = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            k += 1
    return c

def get_initial_vars():
    n = 26
    r_init = 0.09
    dx = 2.0 * r_init
    dy = np.sqrt(3.0) * r_init
    pts = []
    row = 0
    while len(pts) < n:
        y = r_init + row * dy
        if y + r_init > 1.0:
            break
        for col in range(15):
            offset = r_init if row % 2 == 1 else 0.0
            x = r_init + col * dx + offset
            if x + r_init > 1.0:
                break
            pts.append([x, y])
        row += 1
    pts = pts[:n]
    vars_arr = np.zeros(n * 3)
    for i in range(n):
        vars_arr[3*i] = pts[i][0]
        vars_arr[3*i+1] = pts[i][1]
        vars_arr[3*i+2] = r_init
    return vars_arr

def run_packing():
    n = 26
    x0 = get_initial_vars()
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 3000, 'ftol': 1e-9})
        if res.success:
            vars_opt = res.x
        else:
            vars_opt = x0
    except Exception:
        vars_opt = x0
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = vars_opt[3*i]
        centers[i, 1] = vars_opt[3*i+1]
        radii[i] = vars_opt[3*i+2]
        
    return centers, radii, np.sum(radii)
