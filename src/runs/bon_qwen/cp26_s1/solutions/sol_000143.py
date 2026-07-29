# sol_000143 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 68244382) state=522de51c sum of radii=2.587352 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(vars):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[52:])

def cons_boundary(vars):
    """Boundary constraints: circles must stay within [0,1]x[0,1]"""
    c = vars[:52].reshape((26, 2))
    r = vars[52:]
    return np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])

def cons_pairs(vars):
    """Non-overlap constraints: distance >= sum of radii for all pairs"""
    c = vars[:52].reshape((26, 2))
    r = vars[52:]
    vals = np.empty(325)
    idx = 0
    for i in range(26):
        for j in range(i + 1, 26):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            vals[idx] = np.hypot(dx, dy) - r[i] - r[j]
            idx += 1
    return vals

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    centers = []
    counts = [6, 5, 6, 5, 4]  # Sums to 26
    for r_idx, cnt in enumerate(counts):
        for c_idx in range(cnt):
            x = c_idx + (0.5 if r_idx % 2 == 1 else 0)
            y = r_idx * np.sqrt(3) / 2
            centers.append([x, y])
    centers = np.array(centers)
    
    # Scale to [0, 1]
    centers = centers - centers.min(axis=0)
    centers /= centers.max(axis=0)
    
    # Initial radii (feasible starting point)
    r_init = np.full(n, 0.08)
    
    # Flatten variables: [x1, y1, ..., x26, y26, r1, ..., r26]
    x0 = np.concatenate([centers.flatten(), r_init])
    
    # Bounds for all variables
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)] * n
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': cons_boundary},
        {'type': 'ineq', 'fun': cons_pairs}
    ]
    
    # 2. Optimize
    res = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-12})
                   
    final_centers = res.x[:2 * n].reshape((n, 2))
    final_radii = res.x[2 * n:]
    
    # Ensure non-negative radii
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, float(np.sum(final_radii))
