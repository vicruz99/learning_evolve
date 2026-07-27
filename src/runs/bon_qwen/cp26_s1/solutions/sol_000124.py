# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 69804dab) state=b6fb3708 sum of radii=1.299999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def generate_initial_centers(n, seed):
    rng = np.random.default_rng(seed)
    points = []
    s = 0.13  # Initial spacing for hex grid
    for i in range(10):
        for j in range(10):
            x = j * s + (i % 2) * s / 2
            y = i * s * np.sqrt(3) / 2
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                points.append([x, y])
    pts = np.array(points[:n])
    pts += rng.uniform(-0.005, 0.005, size=pts.shape)
    return np.clip(pts, 0.02, 0.98)

def run_packing():
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    for seed in range(8):
        centers0 = generate_initial_centers(n, seed)
        radii0 = np.full(n, 0.05)
        x0 = np.concatenate([centers0.flatten(), radii0])
        
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        def objective(v):
            return -np.sum(v[2*n:])
            
        def constraint_func(v):
            c = v[:2*n].reshape(n, 2)
            r = v[2*n:]
            # Boundary constraints: circle must be inside [0,1]x[0,1]
            cons_b = np.concatenate([
                c[:, 0] - r,
                1 - c[:, 0] - r,
                c[:, 1] - r,
                1 - c[:, 1] - r
            ])
            # Overlap constraints: distance^2 >= (r1 + r2)^2
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist2 = np.sum(diff**2, axis=2)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            # Add small margin to prevent numerical precision violations
            overlap = dist2 - (r_sum + 1e-7)**2
            idx = np.triu_indices(n, k=1)
            cons_o = overlap[idx]
            return np.concatenate([cons_b, cons_o])
            
        cons = NonlinearConstraint(constraint_func, 0, np.inf)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
            curr_sum = -np.sum(res.x[2*n:])
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x[:2*n].reshape(n, 2)
                best_radii = res.x[2*n:]
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = generate_initial_centers(n, 0)
        best_radii = np.full(n, 0.05)
        
    # Safety clipping and tiny shrink to guarantee validator tolerance
    best_radii = np.clip(best_radii, 0, 0.5)
    best_centers = np.clip(best_centers, 1e-9, 1 - 1e-9)
    best_radii *= 0.999999
    
    return best_centers, best_radii, np.sum(best_radii)
