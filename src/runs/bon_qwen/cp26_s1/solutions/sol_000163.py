# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5dc93b19) state=efd7330e sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def evaluate_constraints(vars, n):
    """Vectorized evaluation of boundary and pairwise non-overlap constraints."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_cons = dist_sq[mask] - r_sum[mask]**2
    
    return np.concatenate([b, pair_cons])

def constraint_fun(vars):
    """Wrapper for constraints to satisfy top-level function requirement."""
    return evaluate_constraints(vars, 26)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    def objective(vars):
        return -np.sum(vars[2*n:])
        
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    best_res = None
    best_sum = -np.inf
    
    # Initialization 1: Uniform grid spread
    centers1 = np.zeros((n, 2))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n: break
            centers1[idx, 0] = 0.1 + j * 0.14
            centers1[idx, 1] = 0.1 + i * 0.14
            idx += 1
    radii1 = np.full(n, 0.04)
    x0_1 = np.concatenate([centers1.flatten(), radii1])
    
    # Initialization 2: Staggered hexagonal-like layout
    centers2 = np.zeros((n, 2))
    radii2 = np.full(n, 0.05)
    idx = 0
    for row in range(5):
        y = 0.12 + row * 0.18
        n_cols = 6 if row % 2 == 0 else 5
        start = 0.12 + (0.08 if row % 2 else 0.0)
        end = 0.88 - (0.08 if row % 2 else 0.0)
        xs = np.linspace(start, end, n_cols)
        for x in xs:
            if idx < n:
                centers2[idx] = [x, y]
                idx += 1
    x0_2 = np.concatenate([centers2.flatten(), radii2])
    
    # Optimize from both initializations
    for x0 in [x0_1, x0_2]:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 600, 'ftol': 1e-9})
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_res = res
        except Exception:
            continue
            
    if best_res is not None:
        centers_opt = best_res.x[:2*n].reshape(n, 2)
        radii_opt = best_res.x[2*n:]
        # Ensure non-negativity and valid bounds post-optimization
        radii_opt = np.maximum(radii_opt, 0.0)
        centers_opt = np.clip(centers_opt, 0.0, 1.0)
        return centers_opt, radii_opt, np.sum(radii_opt)
    else:
        return np.zeros((n, 2)), np.zeros(n), 0.0
