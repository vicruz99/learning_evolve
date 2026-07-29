# sol_000356 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c8413f9) state=6dd7dd48 sum of radii=2.579695 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    def objective(vars):
        # Maximize sum of radii => minimize negative sum
        return -np.sum(vars[2::3])
        
    def constraints(vars):
        cx = vars[0::3]
        cy = vars[1::3]
        r = vars[2::3]
        
        c = []
        # Boundary constraints: circles must stay within [0, 1]
        c.append(cx - r)
        c.append(1.0 - cx - r)
        c.append(cy - r)
        c.append(1.0 - cy - r)
        
        # Pairwise non-overlap constraints
        # Compute pairwise distances efficiently using broadcasting
        cx_mat = cx[:, None] - cx[None, :]
        cy_mat = cy[:, None] - cy[None, :]
        dists = np.sqrt(cx_mat**2 + cy_mat**2)
        
        # Extract upper triangle to avoid duplicates and self-comparison
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        rad_sum = r[:, None] + r[None, :]
        c.append(dists[mask] - rad_sum[mask])
        
        return np.concatenate(c)
        
    # Initial layout: Hexagonal-ish grid for high initial density
    centers = np.zeros((N, 2))
    radii = np.ones(N) * 0.05
    idx = 0
    row = 0
    while idx < N:
        # Stagger rows for hexagonal packing effect
        cols = 5 if row < 5 else N - idx
        for col in range(cols):
            centers[idx, 0] = (col + 0.5 + 0.5 * (row % 2)) / 5.0
            centers[idx, 1] = (row + 0.5) / 6.0
            idx += 1
        row += 1
        
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_res = None
    best_sum = -np.inf
    
    # Multiple restarts with jitter to escape local minima
    for seed in range(5):
        np.random.seed(seed + 42)
        x0_jitter = x0.copy()
        x0_jitter[:2*N] += np.random.randn(2*N) * 0.015
        x0_jitter[:2*N] = np.clip(x0_jitter[:2*N], 0.05, 0.95)
        
        res = minimize(objective, x0_jitter, method='SLSQP', bounds=bounds, 
                      constraints=cons, options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False})
        
        current_sum = -res.fun
        if res.success or current_sum > best_sum:
            best_sum = current_sum
            best_res = res
            
    if best_res is not None:
        final_x = best_res.x
        final_centers = np.column_stack((final_x[0::3], final_x[1::3]))
        final_radii = final_x[2::3]
    else:
        # Fallback to initial configuration if optimization fails
        final_centers = centers
        final_radii = radii
        
    # Post-processing to ensure strict validity within numerical tolerances
    final_radii = np.maximum(final_radii, 1e-9)
    final_centers = np.clip(final_centers, final_radii[:, None], 1.0 - final_radii[:, None])
    
    return final_centers, final_radii, np.sum(final_radii)
