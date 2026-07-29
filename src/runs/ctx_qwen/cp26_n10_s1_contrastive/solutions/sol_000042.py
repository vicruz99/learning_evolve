# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=26164787 sum of radii=2.630439 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2*N:])

def constraints(vars):
    """
    Constraint function: ensures circles are inside the unit square and do not overlap.
    Returns a 1D array of constraint values that must be >= 0.
    """
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    c = []
    # Boundary constraints: circle inside [0,1]^2
    # x >= r  => x - r >= 0
    c.append(centers[:, 0] - radii)
    # x <= 1-r => 1 - x - r >= 0
    c.append(1.0 - centers[:, 0] - radii)
    # y >= r  => y - r >= 0
    c.append(centers[:, 1] - radii)
    # y <= 1-r => 1 - y - r >= 0
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints: dist_sq >= (r_i + r_j)^2
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    # Extract only upper triangular pairs to avoid duplicates and self-comparison
    c.append(dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def run_packing():
    # Variable bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Hexagonal-inspired layout (rows: 6, 5, 6, 5, 4)
    rows = [6, 5, 6, 5, 4]
    pts = []
    r_est = 0.09
    y = r_est
    for k, cnt in enumerate(rows):
        x_start = r_est if k % 2 == 0 else 2*r_est
        for m in range(cnt):
            pts.append([x_start + m*2*r_est, y])
        y += np.sqrt(3)*r_est
    pts = np.array(pts[:N])
    
    # Scale to leave room for optimization to expand radii
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    pts_scaled = (pts - mn) / (mx - mn) * 0.8 + 0.1
    r_init = np.full(N, 0.08)
    inits.append(np.concatenate([pts_scaled.flatten(), r_init]))
    
    # 2. 5x5 Grid + 1 center
    g = []
    for i in range(5):
        for j in range(5):
            g.append([0.1 + i*0.2, 0.1 + j*0.2])
    g.append([0.5, 0.5])
    g = np.array(g)
    r_g = np.array([0.09]*25 + [0.02])
    inits.append(np.concatenate([g.flatten(), r_g]))
    
    # 3. Random perturbations of base layouts to escape symmetry/local minima
    np.random.seed(123)
    for _ in range(15):
        c_p = pts_scaled.copy() + np.random.randn(N, 2) * 0.025
        c_p = np.clip(c_p, 0.05, 0.95)
        r_p = r_init.copy() + np.random.randn(N) * 0.015
        r_p = np.clip(r_p, 0.02, 0.25)
        inits.append(np.concatenate([c_p.flatten(), r_p]))
        
        c_p2 = g.copy() + np.random.randn(N, 2) * 0.025
        c_p2 = np.clip(c_p2, 0.05, 0.95)
        r_p2 = r_g.copy() + np.random.randn(N) * 0.01
        r_p2 = np.clip(r_p2, 0.01, 0.2)
        inits.append(np.concatenate([c_p2.flatten(), r_p2]))

    # Run optimization from each initialization
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            # Check if constraints are satisfied within numerical tolerance
            if np.min(constraints(res.x)) >= -1e-8:
                curr_sum = np.sum(res.x[2*N:])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_vars is None:
        best_vars = inits[0]
        
    # Final high-precision refinement on the best configuration found
    res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                         constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
    if np.min(constraints(res_final.x)) >= -1e-8:
        best_vars = res_final.x
        
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    return centers, radii, float(np.sum(radii))
