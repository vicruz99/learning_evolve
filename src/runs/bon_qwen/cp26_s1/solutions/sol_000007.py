# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e6663bde) state=70db8fc3 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_func(vars):
    n = 26
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b1 = vars[0::3] - vars[2::3]
    b2 = 1.0 - vars[0::3] - vars[2::3]
    b3 = vars[1::3] - vars[2::3]
    b4 = 1.0 - vars[1::3] - vars[2::3]
    
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Overlap constraints: dist^2 >= (r1+r2)^2
    overlap_cons = []
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dr = radii[i] + radii[j]
            overlap_cons.append(dx*dx + dy*dy - dr*dr)
            
    return np.concatenate([b1, b2, b3, b4, np.array(overlap_cons)])

def generate_initial_guess(n_circles=26):
    centers = []
    base_x = np.linspace(0.2, 0.8, 5)
    base_y = np.linspace(0.2, 0.8, 5)
    for x in base_x:
        for y in base_y:
            centers.append((x, y))
    centers.extend([(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)])
    np.random.shuffle(centers)
    centers = np.array(centers[:n_circles])
    centers += np.random.uniform(-0.05, 0.05, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n_circles, 0.04)

    # Ensure initial feasibility by shrinking overlapping circles
    for _ in range(100):
        changed = False
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if dist < radii[i] + radii[j]:
                    new_r = dist / 2.0
                    radii[i] = min(radii[i], new_r)
                    radii[j] = min(radii[j], new_r)
                    changed = True
        if not changed:
            break

    vars_init = np.zeros(3 * n_circles)
    for i in range(n_circles):
        vars_init[3*i] = centers[i,0]
        vars_init[3*i+1] = centers[i,1]
        vars_init[3*i+2] = radii[i]
    return vars_init

def run_packing():
    n = 26
    best_x = None
    best_val = -np.inf
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n

    for seed in range(15):
        np.random.seed(seed)
        x0 = generate_initial_guess(n)
        try:
            res = minimize(objective, x0, method='SLSQP',
                           constraints={'type': 'ineq', 'fun': constraint_func},
                           bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun) and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    if best_x is None:
        best_x = generate_initial_guess(n)
        best_val = -objective(best_x)

    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]

    # Final safety clipping to handle numerical drift
    radii = np.clip(radii, 0.0, None)
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(best_val)
