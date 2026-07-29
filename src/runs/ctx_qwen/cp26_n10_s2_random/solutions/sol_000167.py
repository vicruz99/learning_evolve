# sol_000167 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000160 (state 08773110) state=590b18a6 sum of radii=2.618267 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26

def get_lp_matrices(n):
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
    for i in range(n):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1
    return A_ub, pair_indices

A_L, P_IDX = get_lp_matrices(N)

def solve_lp_and_grad(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(A_L.shape[0])
    idx = 0
    for i, j in P_IDX:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_L, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, -1.0, np.zeros_like(centers)
        
    radii = res.x
    duals = res.ineqlin.marginals
    grad = np.zeros_like(centers)
    
    idx = 0
    for i, j in P_IDX:
        lam = duals[idx]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1
        
    b_start = len(P_IDX)
    for i in range(n):
        mu_L = duals[b_start + 4*i]
        mu_R = duals[b_start + 4*i + 1]
        mu_B = duals[b_start + 4*i + 2]
        mu_T = duals[b_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, np.sum(radii), grad

def obj_lp(x):
    _, s, _ = solve_lp_and_grad(x.reshape(N, 2))
    return -s

def generate_starts(rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6], [6, 6, 4, 5, 5], [5, 6, 4, 5, 6],
        [7, 7, 6, 6], [6, 7, 6, 7], [7, 6, 7, 6]
    ]
    for pat in patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105, 0.11]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * math.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for _ in range(5):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        starts.append(c)
        
    return starts

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d)/2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def cons_sl(v):
    cc = v[:2*N].reshape(N, 2)
    rr = v[2*N:]
    con = [cc[:,0]-rr, 1.0-cc[:,0]-rr, cc[:,1]-rr, 1.0-cc[:,1]-rr]
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(cc[idx[0]] - cc[idx[1]], axis=1)
    con.append(d - (rr[idx[0]] + rr[idx[1]]))
    return np.concatenate(con)

def obj_sl(v):
    return -np.sum(v[2*N:])

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: LP Gradient Ascent
    for c0 in starts:
        c = c0.copy()
        step = 0.006
        for k in range(600):
            radii, s, grad = solve_lp_and_grad(c)
            if radii is None: break
            if s > best_sum:
                best_sum = s
                best_c = c.copy()
                best_r = radii.copy()
            gn = np.linalg.norm(grad)
            if gn < 1e-10: break
            c += step * (grad / gn)
            c = np.clip(c, 0.01, 0.99)
            if k % 100 == 0 and k > 0: step *= 0.92
            if k % 250 == 0: c += rng.normal(0, 0.0015, c.shape); c = np.clip(c, 0.02, 0.98)
            
    # Phase 2: SLSQP Joint Refinement
    if best_c is not None:
        bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
        v0 = np.concatenate([best_c.flatten(), best_r])
        for _ in range(3):
            try:
                res = minimize(obj_sl, v0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': cons_sl},
                              options={'maxiter': 4000, 'ftol': 1e-14})
                if np.min(cons_sl(res.x)) >= -1e-7:
                    s = np.sum(res.x[2*N:])
                    if s > best_sum:
                        best_sum = s
                        best_c = res.x[:2*N].reshape(N, 2).copy()
                        best_r = res.x[2*N:].copy()
                        v0 = res.x.copy()
            except Exception: pass
            
        # Phase 3: Powell on Centers with LP Radii
        for _ in range(5):
            c_pert = best_c + rng.normal(0, 0.003, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            try:
                res_p = minimize(obj_lp, c_pert.flatten(), method='Powell', 
                                bounds=[(0,1)]*(2*N), options={'maxiter': 1500, 'ftol': 1e-13})
                c_opt = res_p.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_c = c_opt
                    best_r = r_opt
            except Exception: pass
            
        # Phase 4: Radius Boosting & Re-optimization
        for boost in range(4):
            c_boost = best_c + rng.normal(0, 0.002, best_c.shape)
            c_boost = np.clip(c_boost, 0.02, 0.98)
            try:
                res_b = minimize(obj_lp, c_boost.flatten(), method='Powell',
                                bounds=[(0,1)]*(2*N), options={'maxiter': 800, 'ftol': 1e-13})
                c_b_opt = res_b.x.reshape(N, 2)
                r_b_opt, s_b_opt, _ = solve_lp_and_grad(c_b_opt)
                if s_b_opt > best_sum:
                    best_sum = s_b_opt
                    best_c = c_b_opt
                    best_r = r_b_opt
            except Exception: pass

    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    return centers, radii, float(np.sum(radii))
