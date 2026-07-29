# sol_000292 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e8a963c) state=e7400842 sum of radii=2.630172 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(z):
    radii = z[2::3]
    return -np.sum(radii)

def constraint_func(z):
    vals = []
    for i in range(N_CIRCLES):
        xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
        for j in range(i + 1, N_CIRCLES):
            xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
            vals.append((xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2)
        vals.append(xi - ri)
        vals.append(1.0 - xi - ri)
        vals.append(yi - ri)
        vals.append(1.0 - yi - ri)
    return np.array(vals)

def get_initial_guess():
    r_init = 0.08
    d = 2.0 * r_init
    h = d * np.sqrt(3.0) / 2.0
    centers = []
    radii = []
    # Hexagonal arrangement: 5, 5, 5, 5, 6 circles per row
    row_counts = [5, 5, 5, 5, 6]
    for k, cnt in enumerate(row_counts):
        y = r_init + k * h
        for m in range(cnt):
            if k % 2 == 0:
                x = r_init + m * d
            else:
                x = r_init + d / 2.0 + m * d
            centers.append([x, y])
            radii.append(r_init)
    z = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        z[3*i] = centers[i][0]
        z[3*i+1] = centers[i][1]
        z[3*i+2] = radii[i]
    return z

def run_packing():
    z0 = get_initial_guess()
    bounds = [(0.0, 1.0) for _ in range(3 * N_CIRCLES)]
    for i in range(N_CIRCLES):
        bounds[3*i+2] = (0.0, 0.5)
        
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
    
    z_opt = res.x
    centers = np.array([[z_opt[3*i], z_opt[3*i+1]] for i in range(N_CIRCLES)])
    radii = np.array([z_opt[3*i+2] for i in range(N_CIRCLES)])
    
    # Ensure strict feasibility due to numerical precision
    radii = np.maximum(radii, 0.0)
    for i in range(N_CIRCLES):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    return centers, radii, float(np.sum(radii))
