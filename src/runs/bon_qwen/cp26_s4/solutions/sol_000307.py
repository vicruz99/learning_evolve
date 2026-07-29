# sol_000307 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e3d19f45) state=32c0b8d9 sum of radii=2.210000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars_):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars_[2 * N_CIRCLES:])

def constraints(vars_):
    """Inequality constraints: boundary and pairwise non-overlap"""
    c = vars_[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = vars_[2 * N_CIRCLES:]
    
    # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
    b = np.concatenate([
        c[:, 0] - r,
        1 - c[:, 0] - r,
        c[:, 1] - r,
        1 - c[:, 1] - r
    ])
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Strictly lower triangle indices for unique pairs
    i, j = np.tril_indices(N_CIRCLES, -1)
    pairs = dist_sq[i, j] - r_sum[i, j]**2
    
    return np.concatenate([b, pairs])

def run_packing():
    # 1. Hexagonal lattice initialization
    r_init = 0.085
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, r_init)
    
    idx = 0
    y = r_init
    row = 0
    while idx < N_CIRCLES:
        x = r_init + (row % 2) * r_init
        while x + r_init <= 1.0 and idx < N_CIRCLES:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
        
    x0 = np.concatenate([centers.ravel(), radii])
    
    # 2. Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 0.5)] * N_CIRCLES
    
    # 3. Optimization setup
    cons = {'type': 'ineq', 'fun': constraints}
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        
        if res.success:
            final_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
            final_radii = res.x[2 * N_CIRCLES:]
        else:
            # Fallback to initial configuration if solver fails
            final_centers = x0[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
            final_radii = x0[2 * N_CIRCLES:]
            
    except Exception:
        # Ultimate fallback
        final_centers = x0[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
        final_radii = x0[2 * N_CIRCLES:]
        
    # 4. Post-processing: ensure strict validity
    final_radii = np.maximum(final_radii, 1e-7)
    final_centers = np.clip(final_centers, final_radii[:, None], 1.0 - final_radii[:, None])
    
    return final_centers, final_radii, np.sum(final_radii)
