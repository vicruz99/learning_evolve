# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2296af5d) state=5c53ffc2 sum of radii=2.607618 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    # Maximize sum of radii => minimize negative sum
    return -np.sum(x[2*N:])

def constraint_func(x):
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    con = []
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    # Pairwise constraints: dist^2 - (r_i + r_j)^2 >= 0
    diffs = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    r_sum_sq = (r[:, None] + r[None, :])**2
    
    # Only upper triangle to avoid duplicate constraints
    idx = np.triu_indices(N, k=1)
    con.append(dist_sq[idx] - r_sum_sq[idx])
    
    return np.concatenate(con).flatten()

def run_packing():
    # Feasible initial configuration: perturbed grid with small radii
    np.random.seed(42)
    cols = np.linspace(0.125, 0.875, 5)
    rows = np.linspace(0.125, 0.875, 5)
    centers = np.array([(c, r) for c in cols for r in rows])
    centers = np.vstack([centers, [0.5, 0.5]])
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    radii = np.full(N, 0.05)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0) for _ in range(2*N)] + [(0.001, 0.5) for _ in range(N)]
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
    
    final_centers = res.x[:2*N].reshape(N, 2)
    final_radii = res.x[2*N:]
    
    # Post-processing to strictly satisfy constraints within tolerance
    for _ in range(5):
        changed = False
        # Boundary check
        for i in range(N):
            cx, cy = final_centers[i]
            r = final_radii[i]
            lim = min(cx, 1-cx, cy, 1-cy)
            if r > lim + 1e-14:
                final_radii[i] = max(0.0, lim)
                changed = True
                
        # Pairwise check
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                sum_r = final_radii[i] + final_radii[j]
                if sum_r > d + 1e-14:
                    excess = sum_r - d
                    if final_radii[i] >= final_radii[j]:
                        final_radii[i] -= excess
                    else:
                        final_radii[j] -= excess
                    final_radii[i] = max(0.0, final_radii[i])
                    final_radii[j] = max(0.0, final_radii[j])
                    changed = True
        if not changed:
            break
            
    return final_centers, final_radii, float(np.sum(final_radii))
