# sol_000221 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000216 (state 64a1292d) state=610e9619 sum of radii=2.194287 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N_CIRCLES = 26
PAIR_I, PAIR_J = np.triu_indices(N_CIRCLES, k=1)
N_PAIRS = len(PAIR_I)
A_UB = np.zeros((N_PAIRS, N_CIRCLES))
for k, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
    A_UB[k, i] = 1.0
    A_UB[k, j] = 1.0

def get_lp_data(centers):
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(lim, 1e-9)) for lim in limits]
    diffs = centers[PAIR_I] - centers[PAIR_J]
    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
    return bounds, b_ub

def lp_solve(centers):
    bounds, b_ub = get_lp_data(centers)
    res = linprog(-np.ones(N_CIRCLES), A_ub=A_UB, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success or not np.isfinite(res.fun):
        return None, None, None
        
    radii = res.x
    sum_r = -res.fun
    
    grad_c = np.zeros((N_CIRCLES, 2))
    try:
        marginals = res.ineqlin.marginals
    except AttributeError:
        try:
            marginals = res.marginals.ineqlin
        except AttributeError:
            marginals = None
            
    if marginals is not None:
        active = marginals > 1e-8
        if np.any(active):
            k_idx = np.where(active)[0]
            i_a = PAIR_I[k_idx]
            j_a = PAIR_J[k_idx]
            d_a = b_ub[k_idx]
            lam = marginals[k_idx]
            diff_a = centers[i_a] - centers[j_a]
            inv_d = 1.0 / np.maximum(d_a, 1e-12)
            vec = diff_a * inv_d[:, np.newaxis] * lam[:, np.newaxis]
            for idx in range(len(k_idx)):
                grad_c[i_a[idx]] += vec[idx]
                grad_c[j_a[idx]] -= vec[idx]
                
    return sum_r, radii, grad_c

def obj_and_grad(x):
    c = x.reshape(N_CIRCLES, 2)
    s, r, g = lp_solve(c)
    if g is None:
        return 1e6, np.zeros_like(x)
    return -s, g.flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    inits = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4], 
        [6,6,5,5,4], [5,4,6,6,5], [5,6,4,6,5], [6,5,5,6,4],
        [5,5,5,5,6], [7,5,5,5,4], [4,7,5,5,5], [6,4,6,5,5]
    ]
    
    np.random.seed(42)
    for pat in patterns:
        if sum(pat) < n: continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n: pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        pts = np.array(pts[:n]) + np.random.uniform(-0.02, 0.02, (n, 2))
        pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts.flatten())
        
    for _ in range(15):
        inits.append(np.random.uniform(0.1, 0.9, (n, 2)).flatten())
        
    bounds_centers = [(0.005, 0.995)] * (2 * n)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: L-BFGS-B from multiple diverse starts
    for x0 in inits:
        try:
            res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds_centers,
                           options={'maxiter': 3000, 'ftol': 1e-15})
            c_opt = res.x.reshape(n, 2)
            s, r, _ = lp_solve(c_opt)
            if s is not None and s > best_sum:
                best_sum = s
                best_centers = c_opt.copy()
                best_radii = r.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation & Restart to escape local minima
    if best_centers is not None:
        x_curr = best_centers.flatten()
        for step in range(40):
            pert_size = 0.012 * (0.92 ** step)
            x_pert = x_curr + np.random.uniform(-pert_size, pert_size, 2*n)
            x_pert = np.clip(x_pert, 0.005, 0.995)
            
            try:
                res_p = minimize(obj_and_grad, x_pert, method='L-BFGS-B', jac=True, bounds=bounds_centers,
                                 options={'maxiter': 2500, 'ftol': 1e-14})
                if np.isfinite(res_p.fun):
                    c_p = res_p.x.reshape(n, 2)
                    s_p, r_p, _ = lp_solve(c_p)
                    if s_p is not None and s_p > best_sum + 1e-7:
                        best_sum = s_p
                        best_centers = c_p.copy()
                        best_radii = r_p.copy()
                        x_curr = res_p.x
            except Exception:
                continue
                
    # Phase 3: Strict Safety Scaling to guarantee numerical validity
    if best_centers is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
                
        diffs = best_centers[PAIR_I] - best_centers[PAIR_J]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        for k in range(N_PAIRS):
            rs = best_radii[PAIR_I[k]] + best_radii[PAIR_J[k]]
            if rs > 1e-12:
                scale = min(scale, dists[k] / rs)
                
        best_radii *= scale * 0.9999995
        best_sum = float(np.sum(best_radii))
    else:
        best_centers = np.random.uniform(0.1, 0.9, (n, 2))
        best_radii = np.full(n, 0.05)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
