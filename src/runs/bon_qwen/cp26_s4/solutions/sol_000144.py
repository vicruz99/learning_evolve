# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 466799c7) state=e5b4ec74 sum of radii=2.591785 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(p):
    """Minimize negative sum of radii."""
    return -np.sum(p[2::3])

def constraint_func(p):
    """Return array of constraint values >= 0."""
    centers = p.reshape(-1, 3)[:, :2]
    radii = p[2::3]
    
    c = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Non-overlap constraints: dist_ij - (r_i + r_j) >= 0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract lower triangular part (i < j)
    mask = np.tril(np.ones((N, N), dtype=bool), k=-1)
    c.append(dist[mask] - sum_r[mask])
    
    return np.concatenate(c)

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    p0_list = []
    
    # 1. Hexagonal lattice initialization (dense, feasible)
    r0 = 0.07
    centers = []
    idx = 0
    row = 0
    while idx < N:
        y = r0 + row * np.sqrt(3) * r0
        col = 0
        while y <= 1.0 - r0 and idx < N:
            x = r0 + col * 2 * r0 + (row % 2) * r0
            if x <= 1.0 - r0:
                centers.append([x, y])
                idx += 1
                col += 1
            else:
                break
        row += 1
    if len(centers) >= N:
        c = np.array(centers[:N])
        p = np.zeros(3 * N)
        p[0::3] = c[:, 0]
        p[1::3] = c[:, 1]
        p[2::3] = r0
        p0_list.append(p)
        
    # 2. Random initializations with small radii to ensure initial feasibility
    np.random.seed(42)
    for _ in range(4):
        c = np.random.uniform(0.15, 0.85, (N, 2))
        p = np.zeros(3 * N)
        p[0::3] = c[:, 0]
        p[1::3] = c[:, 1]
        p[2::3] = 0.02  # Small radius guarantees initial feasibility
        p0_list.append(p)
        
    # Optimization runs
    for p0 in p0_list:
        res = minimize(objective, p0, method='SLSQP', 
                       constraints={'type': 'ineq', 'fun': constraint_func},
                       options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False})
        
        if np.isfinite(res.fun):
            cur_sum = -res.fun
            centers_cand = res.x.reshape(-1, 3)[:, :2]
            radii_cand = res.x[2::3]
            
            # Feasibility check
            if np.all(radii_cand >= 0) and np.all(centers_cand >= 0) and np.all(centers_cand <= 1):
                diff = centers_cand[:, None, :] - centers_cand[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                sums_r = radii_cand[:, None] + radii_cand[None, :]
                np.fill_diagonal(dists, np.inf)
                
                # Check against validation tolerance
                if np.all(dists >= sums_r - 1e-11):
                    if cur_sum > best_sum:
                        best_sum = cur_sum
                        best_centers = centers_cand.copy()
                        best_radii = radii_cand.copy()
                        
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (N, 2))
        best_radii = np.full(N, 0.01)
        best_sum = np.sum(best_radii)
        
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
