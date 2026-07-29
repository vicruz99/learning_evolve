# sol_000218 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=24616ab3 sum of radii=2.583440 correctness=1.0
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

def gradient_ascent(centers0, max_iter=3000, init_step=0.006, rng=None):
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = init_step
    
    radii, curr_sum, duals = solve_lp_radii(centers)
    best_sum = curr_sum
    
    for k in range(max_iter):
        grad = compute_gradient(centers, duals)
        g_norm = np.linalg.norm(grad)
        
        if g_norm < 1e-10:
            break
            
        centers_new = centers + step * (grad / g_norm)
        centers_new = np.clip(centers_new, 1e-6, 1.0 - 1e-6)
        
        radii_new, sum_new, duals_new = solve_lp_radii(centers_new)
        
        if sum_new > curr_sum:
            centers = centers_new
            curr_sum = sum_new
            duals = duals_new
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = centers.copy()
            step = min(step * 1.15, 0.025)
        else:
            step *= 0.6
            if step < 1e-10:
                break
                
        if rng is not None and k > 0 and k % 250 == 0:
            noise_scale = 0.003 * (0.7 ** (k // 250))
            centers += rng.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
            radii, curr_sum, duals = solve_lp_radii(centers)
            
    return best_centers, best_sum

def generate_starts(rng):
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5],
        [5, 6, 4, 5, 6], [6, 4, 6, 5, 5], [4, 5, 6, 5, 6],
        [6, 5, 4, 5, 6], [5, 5, 6, 6, 4]
    ]
    for pat in patterns:
        for r0 in [0.088, 0.092, 0.096, 0.100, 0.104]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.array(c[:N]))
            
    for _ in range(30):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i]-c[j])
                    if d < 0.22 and d > 1e-5:
                        f = (0.22 - d) / d
                        diff = c[i] - c[j]
                        forces[i] += diff * f
                        forces[j] -= diff * f
            c += forces * 0.005
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
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

def slsqp_optimize(c_init, r_init, maxiter=8000):
    v0 = np.concatenate([c_init.flatten(), r_init])
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=SLSQP_BOUNDS,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c_init, r_init, 0.0

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(200):
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
    rng = np.random.default_rng(123)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient ascent on diverse starts
    for c_init in starts:
        r_lp, s_lp, _ = solve_lp_radii(c_init)
        if s_lp <= best_sum:
            continue
            
        c_opt, s_opt = gradient_ascent(c_init, max_iter=3500, init_step=0.007, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            r_lp, _, _ = solve_lp_radii(best_c)
            best_r = r_lp.copy()
            
    # Phase 2: SLSQP joint polish
    if best_c is not None:
        r_init = np.minimum(np.minimum(best_c[:, 0], 1.0 - best_c[:, 0]),
                            np.minimum(best_c[:, 1], 1.0 - best_c[:, 1]))
        dists = np.linalg.norm(best_c[:, None, :] - best_c[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(r_init, rp) * 0.92
        
        c_slsqp, r_slsqp, s_slsqp = slsqp_optimize(best_c, r_init, maxiter=10000)
        if s_slsqp > best_sum:
            best_sum = s_slsqp
            best_c = c_slsqp.copy()
            best_r = r_slsqp.copy()
            
        # Phase 3: Perturbation & Re-optimization loop
        for _ in range(12):
            c_per = best_c + rng.normal(0, 0.004, best_c.shape)
            c_per = np.clip(c_per, 0.02, 0.98)
            
            r_lp_per, _, _ = solve_lp_radii(c_per)
            c_ga, s_ga = gradient_ascent(c_per, max_iter=2000, init_step=0.005, rng=rng)
            
            if s_ga > best_sum:
                best_sum = s_ga
                best_c = c_ga.copy()
                r_lp, _, _ = solve_lp_radii(best_c)
                best_r = r_lp.copy()
                
            r_init_loc = np.minimum(np.minimum(c_per[:, 0], 1.0 - c_per[:, 0]),
                                    np.minimum(c_per[:, 1], 1.0 - c_per[:, 1]))
            dists_loc = np.linalg.norm(c_per[:, None, :] - c_per[None, :, :], axis=2)
            np.fill_diagonal(dists_loc, np.inf)
            rp_loc = 0.5 * np.min(dists_loc, axis=1)
            r_init_loc = np.minimum(r_init_loc, rp_loc) * 0.92
            
            c_loc, r_loc, s_loc = slsqp_optimize(c_per, r_init_loc, maxiter=5000)
            if s_loc > best_sum:
                best_sum = s_loc
                best_c = c_loc.copy()
                best_r = r_loc.copy()
                
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
