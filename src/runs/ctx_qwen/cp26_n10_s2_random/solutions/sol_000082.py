# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000058 (state f7fedeb3) state=c7582b14 sum of radii=2.603492 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution

N_CIRCLES = 26
# Precompute triangular indices for pairwise constraints
TRIU_INDICES = np.triu_indices(N_CIRCLES, k=1)

def fast_sum_radii(centers):
    """
    Computes a feasible sum of radii for fixed centers.
    r_i = min(distance_to_boundary, 0.5 * min_distance_to_other_circles)
    """
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    p = 0.5 * np.min(d, axis=1)
    return np.sum(np.minimum(b, p))

def obj_centers(x):
    """Objective for center optimization: minimize negative sum of radii."""
    return -fast_sum_radii(x.reshape(N_CIRCLES, 2))

def compute_slsqp_obj(v):
    """Objective for joint SLSQP optimization."""
    return -np.sum(v[2*N_CIRCLES:])

def compute_slsqp_cons(v):
    """Computes inequality constraints for SLSQP (must be >= 0)."""
    c = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2*N_CIRCLES:]
    out = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    
    d2 = (c[:, np.newaxis, :] - c[np.newaxis, :, :])**2
    dists = np.sqrt(np.sum(d2, axis=2))
    np.fill_diagonal(dists, np.inf)
    out.append(dists[TRIU_INDICES] - (r[TRIU_INDICES[0]] + r[TRIU_INDICES[1]]))
    return np.concatenate(out)

def solve_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers."""
    n_pairs = N_CIRCLES*(N_CIRCLES-1)//2
    n_bnd = 4*N_CIRCLES
    A_ub = np.zeros((n_pairs+n_bnd, N_CIRCLES))
    idx = 0
    for i, j in zip(TRIU_INDICES[0], TRIU_INDICES[1]):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        idx += 1
    for i in range(N_CIRCLES):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
            
    b_ub = np.zeros(n_pairs+n_bnd)
    idx = 0
    for i, j in zip(TRIU_INDICES[0], TRIU_INDICES[1]):
        b_ub[idx] = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
        idx += 1
    for i in range(N_CIRCLES):
        x, y = centers[i]
        b_ub[idx]=x; idx+=1
        b_ub[idx]=1.0-x; idx+=1
        b_ub[idx]=y; idx+=1
        b_ub[idx]=1.0-y; idx+=1
        
    res = linprog(-np.ones(N_CIRCLES), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    return res.x if res.success else np.full(N_CIRCLES, 0.08)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    
    # 1. Global search with Differential Evolution on centers
    bounds_de = [(0.05, 0.95)] * (2 * N_CIRCLES)
    res_de = differential_evolution(obj_centers, bounds_de, popsize=20, maxiter=150, 
                                    seed=42, workers=1, tol=1e-7, polish=False)
    best_centers = res_de.x.reshape(N_CIRCLES, 2)
    
    # 2. Local refinement with Powell's method
    res_powell = minimize(obj_centers, best_centers.flatten(), method='Powell', 
                          options={'maxiter': 1500, 'ftol': 1e-10, 'xtol': 1e-10})
    best_centers = res_powell.x.reshape(N_CIRCLES, 2)
    
    # 3. Compute exact optimal radii via Linear Programming
    best_radii = solve_lp_radii(best_centers)
    
    # 4. Joint SLSQP polish (centers + radii) for final micro-adjustments
    x0 = np.concatenate([best_centers.flatten(), best_radii])
    bounds_slsqp = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    try:
        res = minimize(compute_slsqp_obj, x0, method='SLSQP', bounds=bounds_slsqp, 
                       constraints={'type':'ineq', 'fun': compute_slsqp_cons},
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
        if -res.fun > np.sum(best_radii) - 1e-6:
            best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
            best_radii = res.x[2*N_CIRCLES:]
    except Exception:
        pass
        
    # 5. Final safety shrink to guarantee validation tolerance compliance
    for _ in range(10):
        changed = False
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < best_radii[i]+best_radii[j]-1e-12:
                    sh = (best_radii[i]+best_radii[j]-d)/2.0 + 1e-9
                    best_radii[i] -= sh
                    best_radii[j] -= sh
                    changed = True
        for i in range(N_CIRCLES):
            mx = min(best_centers[i,0], 1.0-best_centers[i,0], best_centers[i,1], 1.0-best_centers[i,1])
            if best_radii[i] > mx + 1e-12:
                best_radii[i] = mx
                changed = True
        if not changed:
            break
            
    return best_centers, best_radii, float(np.sum(best_radii))
