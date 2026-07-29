# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34f92e2c) state=aac2881c sum of radii=2.066859 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    best_val = -np.inf
    best_x = None

    # Try multiple starts around a hexagonal lattice to escape local minima
    for seed in range(5):
        np.random.seed(seed)
        pts = []
        y = 0.0
        for i in range(6):
            for j in range(5):
                if len(pts) >= n: break
                x = j + (0.5 if i % 2 else 0.0)
                pts.append([x, y])
            y += np.sqrt(3)/2

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        
        # Normalize to [0.1, 0.9] range with some margin
        xs = (xs - xs.min()) / (xs.max() - xs.min()) * 0.7 + 0.15
        ys = (ys - ys.min()) / (ys.max() - ys.min()) * 0.7 + 0.15

        # Add small random perturbation
        xs += np.random.uniform(-0.02, 0.02, n)
        ys += np.random.uniform(-0.02, 0.02, n)
        centers = np.column_stack([xs, ys])
        r0 = np.full(n, 0.03)
        v0 = np.concatenate([centers.ravel(), r0])

        def obj(v):
            return -np.sum(v[2*n:])

        cons = []
        for i in range(n):
            # x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n+i]})
            # 1 - x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[2*n+i]})
            # y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n+i]})
            # 1 - y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[2*n+i]})

        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j:
                    (v[2*i]-v[2*j])**2 + (v[2*i+1]-v[2*j+1])**2 - (v[2*n+i]+v[2*n+j])**2})

        bnds = [(0, 1)] * (2*n) + [(1e-4, 0.5)] * n

        try:
            res = minimize(obj, v0, method='SLSQP', bounds=bnds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-9, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        # Fallback configuration (should not be reached)
        centers = np.column_stack([np.linspace(0.1, 0.9, 6) for _ in range(5)][:26]).reshape(26, 2)
        radii = np.full(26, 0.08)
        return centers, radii, np.sum(radii)

    centers = best_x[:2*n].reshape((n, 2))
    radii = best_x[2*n:]

    # Post-processing to strictly satisfy validation tolerances
    # Clamp to boundaries
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if radii[i] > max_r + 1e-12:
            radii[i] = max_r - 1e-9

    # Clamp to pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < radii[i] + radii[j] - 1e-12:
                shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)

    return centers, radii, np.sum(radii)
