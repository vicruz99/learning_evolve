# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000160 (state 08773110) state=79899e79 sum of radii=2.624553 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_IND = np.triu_indices(N, 1)

def build_lp_structure(n):
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A = np.zeros((num_pairs + num_bound, n))
    pair_idx = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            pair_idx.append((i, j))
            k += 1
    for i in range(n):
        base = num_pairs + 4 * i
        A[base, i] = 1.0
        A[base + 1, i] = 1.0
        A[base + 2, i] = 1.0
        A[base + 3, i] = 1.0
    return A, pair_idx

A_LP, PAIR_IDX = build_lp_structure(N)

def solve_lp_radii(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    idx = 0
    for i, j in PAIR_IDX:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        return res.x, np.sum(res.x), res.ineqlin.marginals
    return np.zeros(n), 0.0, np.zeros_like(b)

def compute_gradient(centers, duals):
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    idx = 0
    for i, j in PAIR_IDX:
        mu = duals[idx]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(n):
        mu_L = duals[bound_start + 4*i]
        mu_R = duals[bound_start + 4*i + 1]
        mu_B = duals[bound_start + 4*i + 2]
        mu_T = duals[bound_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return grad

def gradient_ascent(centers0, max_iter=2500, init_step=0.006):
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    no_improve = 0
    
    for k in range(max_iter):
        radii, curr_sum, duals = solve_lp_radii(centers)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 100:
            step *= 0.65
        elif no_improve > 40:
            step *= 0.85
            
        if step < 1e-11:
            break
            
        grad = compute_gradient(centers, duals)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        centers += step * (grad / grad_norm)
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
        if k % 250 == 0 and k > 0:
            centers += np.random.normal(0, step * 0.15, centers.shape)
            centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
            
    return best_centers, best_sum

def generate_starts(rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5]
    ]
    for pat in patterns:
        for r0 in [0.088, 0.092, 0.096, 0.10, 0.104]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.array(c[:N]))
            
    for _ in range(25):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    return starts

def slsqp_obj(v):
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_IND[0], 0] - c[TRIU_IND[1], 0]
    dy = c[TRIU_IND[0], 1] - c[TRIU_IND[1], 1]
    dr = r[TRIU_IND[0]] + r[TRIU_IND[1]]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

SLSQP_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def slsqp_optimize(c_init, r_init):
    v0 = np.concatenate([c_init.flatten(), r_init])
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=SLSQP_BOUNDS,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 6000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c_init, r_init, 0.0

def repair(centers, radii):
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    for c_init in starts:
        ub = np.minimum(np.minimum(c_init[:, 0], 1.0 - c_init[:, 0]),
                        np.minimum(c_init[:, 1], 1.0 - c_init[:, 1]))
        dists = np.linalg.norm(c_init[:, None, :] - c_init[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(ub, rp) * 0.85
        
        c_opt, r_opt, s_opt = slsqp_optimize(c_init, r_init)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    if best_c is not None:
        r_lp, s_lp, _ = solve_lp_radii(best_c)
        if s_lp > best_sum:
            best_sum = s_lp
            best_r = r_lp
            
        for _ in range(5):
            c_ga, s_ga = gradient_ascent(best_c, max_iter=3500, init_step=0.007)
            if s_ga > best_sum:
                best_sum = s_ga
                best_c = c_ga
                r_lp, _, _ = solve_lp_radii(best_c)
                best_r = r_lp
                
        for _ in range(8):
            c_per = best_c + rng.normal(0, 0.003, best_c.shape)
            c_per = np.clip(c_per, 0.02, 0.98)
            r_per, _, _ = solve_lp_radii(c_per)
            c_opt, r_opt, s_opt = slsqp_optimize(c_per, r_per)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt
                best_r = r_opt
                
        r_final, s_final, _ = solve_lp_radii(best_c)
        if s_final > best_sum:
            best_sum = s_final
            best_r = r_final
            
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
