# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e52471dd) state=91e39415 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = 0.0
    best_vars = None
    
    np.random.seed(42)
    
    for restart in range(4):
        # Initialize centers in a perturbed grid pattern
        centers = np.array([(0.1 + i * 0.2, 0.1 + j * 0.2) for i in range(5) for j in range(5)])
        centers = np.vstack([centers, [0.5, 0.55]]) # Add 26th circle
        centers += np.random.uniform(-0.015, 0.015, centers.shape)
        radii = np.full(N, 0.085)
        
        vars0 = np.zeros(3 * N)
        vars0[::3] = centers[:, 0]
        vars0[1::3] = centers[:, 1]
        vars0[2::3] = radii
        
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
        
        try:
            res = minimize(_objective, vars0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': _constraint_func},
                           options={'maxiter': 2000, 'ftol': 1e-12})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x
        except Exception:
            continue
            
    if best_vars is not None:
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        centers[:, 0] = best_vars[::3]
        centers[:, 1] = best_vars[1::3]
        radii = best_vars[2::3]
        
        # Ensure strict feasibility with boundaries
        radii = np.clip(radii, 1e-6, None)
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
        # Iteratively resolve minor overlaps if any remain due to numerical precision
        for _ in range(10):
            max_ov = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    req = radii[i] + radii[j]
                    if d < req:
                        ov = req - d
                        if ov > max_ov:
                            max_ov = ov
            if max_ov < 1e-9:
                break
            radii *= (1.0 - max_ov * 0.5)
            
        return centers, radii, np.sum(radii)
        
    return np.zeros((N, 2)), np.zeros(N), 0.0

def _objective(vars):
    return -np.sum(vars[2::3])

def _constraint_func(vars):
    c = vars[:2 * N].reshape(N, 2)
    r = vars[2::3]
    cons = []
    
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(dx * dx + dy * dy - (r[i] + r[j]) ** 2)
            
    return np.concatenate(cons)
