# sol_000272 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000237 (state 963256f0) state=f361d239 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute constant LP constraint matrix structure for pairwise distances
A_LP = np.zeros((N * (N - 1) // 2, N))
PAIR_INDICES = []
_lp_row = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[_lp_row, i] = 1.0
        A_LP[_lp_row, j] = 1.0
        PAIR_INDICES.append((i, j))
        _lp_row += 1

def solve_lp_and_gradient(centers):
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    b_ub = dists[np.triu_indices(n, k=1)]
    
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    duals_ineq = np.zeros(A_LP.shape[0])
    if hasattr(res, 'marginals') and res.marginals is not None:
        if hasattr(res.marginals, 'ineqlin'):
            duals_ineq = res.marginals.ineqlin
        elif hasattr(res.marginals, 'ineq'):
            duals_ineq = res.marginals.ineq
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals_ineq = res.ineqlin.marginals
        
    dual_mat = np.zeros((n, n))
    idx = 0
    for i, j in PAIR_INDICES:
        val = duals_ineq[idx]
        if val > 1e-10:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                dual_mat[i, j] = val
                dual_mat[j, i] = -val
        idx += 1
        
    unit_diff = diff / np.maximum(dists, 1e-9)[..., np.newaxis]
    grad = np.einsum('ij,ijk->ik', dual_mat, unit_diff)
    
    return radii, np.sum(radii), grad

def obj_grad_lbfgs(x_flat):
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    radii, val, grad = solve_lp_and_gradient(centers)
    return -val, -grad.flatten()

def generate_inits(rng):
    configs = []
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [6, 5, 5, 5, 5], [5, 6, 4, 5, 6], [5, 5, 4, 6, 6]
    ]
    for pat in patterns:
        c = []
        r0 = 0.10
        y = r0
        for r_idx, cnt in enumerate(pat):
            shift = r0 if r_idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(c) < N:
                    c.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3.0)
        while len(c) < N:
            c.append(rng.uniform(0.1, 0.9, 2))
        c = np.array(c[:N])
        for _ in range(4):
            c_pert = c + rng.normal(0, 0.005, c.shape)
            c_pert = np.clip(c_pert, 0.05, 0.95)
            configs.append(c_pert)
            
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        configs.append(c)
        
    return configs

def optimize_lbfgs(c0, bounds):
    try:
        res = minimize(obj_grad_lbfgs, c0.flatten(), jac=True, method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2)
    except Exception:
        return c0

def polish_slsqp(centers, radii):
    n = centers.shape[0]
    v0 = np.concatenate([centers.flatten(), radii])
    
    def obj(v):
        return -np.sum(v[2 * n:])
        
    def cons(v):
        cc = v[:2 * n].reshape(n, 2)
        rr = v[2 * n:]
        c_list = [
            cc[:, 0] - rr,
            1.0 - cc[:, 0] - rr,
            cc[:, 1] - rr,
            1.0 - cc[:, 1] - rr
        ]
        idx = np.triu_indices(n, 1)
        d = np.linalg.norm(cc[idx[0]] - cc[idx[1]], axis=1)
        c_list.append(d - (rr[idx[0]] + rr[idx[1]]))
        return np.concatenate(c_list)
        
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-9:
            return res.x[:2 * n].reshape(n, 2), res.x[2 * n:], np.sum(res.x[2 * n:])
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def repair_packing(centers, radii):
    radii = radii.copy()
    n = centers.shape[0]
    for _ in range(80):
        changed = False
        for i in range(n):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_xy = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    configs = generate_inits(rng)
    
    for c0 in configs:
        c_opt = optimize_lbfgs(c0, bounds_xy)
        radii, s_opt, _ = solve_lp_and_gradient(c_opt)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = radii.copy()
            
    for _ in range(40):
        c_pert = best_centers.copy()
        idx = rng.choice(N, size=5, replace=False)
        c_pert[idx] += rng.normal(0, 0.03, (5, 2))
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        c_opt = optimize_lbfgs(c_pert, bounds_xy)
        radii, s_opt, _ = solve_lp_and_gradient(c_opt)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = radii.copy()
            
    c_pol, r_pol, s_pol = polish_slsqp(best_centers, best_radii)
    if s_pol > best_sum:
        best_sum = s_pol
        best_centers = c_pol
        best_radii = r_pol
        
    for _ in range(30):
        c_pert = best_centers.copy()
        idx = rng.choice(N, size=3, replace=False)
        c_pert[idx] += rng.normal(0, 0.015, (3, 2))
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        c_opt = optimize_lbfgs(c_pert, bounds_xy)
        radii, s_opt, _ = solve_lp_and_gradient(c_opt)
        
        c_pol, r_pol, s_pol = polish_slsqp(c_opt, radii)
        if s_pol > best_sum:
            best_sum = s_pol
            best_centers = c_pol
            best_radii = r_pol
            
    best_radii = repair_packing(best_centers, best_radii)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
