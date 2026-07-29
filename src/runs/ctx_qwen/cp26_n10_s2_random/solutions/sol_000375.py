# sol_000375 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000303 (state 682ce44f) state=d00ea5e3 sum of radii=2.630957 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_I = np.zeros(NUM_PAIRS, dtype=int)
PAIR_J = np.zeros(NUM_PAIRS, dtype=int)
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_I[idx] = i
        PAIR_J[idx] = j
        idx += 1
for i in range(N):
    A_LP[idx + 4*i, i] = 1.0
    A_LP[idx + 4*i + 1, i] = 1.0
    A_LP[idx + 4*i + 2, i] = 1.0
    A_LP[idx + 4*i + 3, i] = 1.0

def solve_lp_and_grad(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(NUM_PAIRS + NUM_BOUND)
    b_ub[:NUM_PAIRS] = dists[np.triu_indices(n, 1)]
    for i in range(n):
        b_ub[NUM_PAIRS + 4*i] = centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 1] = 1.0 - centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 2] = centers[i, 1]
        b_ub[NUM_PAIRS + 4*i + 3] = 1.0 - centers[i, 1]
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
            
        radii = res.x
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            try:
                duals = np.asarray(res.ineqlin.marginals)
            except Exception:
                duals = np.zeros(len(b_ub))
                
        grad = np.zeros_like(centers)
        active = duals[:NUM_PAIRS] > 1e-9
        if np.any(active):
            i_idx = PAIR_I[active]
            j_idx = PAIR_J[active]
            d = dists[i_idx, j_idx]
            safe_d = np.where(d > 1e-9, d, 1e-9)
            vec = (centers[i_idx] - centers[j_idx]) / safe_d[:, np.newaxis]
            lam = duals[:NUM_PAIRS][active][:, np.newaxis]
            grad[i_idx] += vec * lam
            grad[j_idx] -= vec * lam
            
        idx_base = NUM_PAIRS + 4 * np.arange(n)
        grad[:, 0] += duals[idx_base] - duals[idx_base + 1]
        grad[:, 1] += duals[idx_base + 2] - duals[idx_base + 3]
            
        return radii, np.sum(radii), grad
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)

def obj_grad(x):
    c = np.clip(x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def slsqp_obj(v):
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def slsqp_polish(c0, r0):
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_inits(rng):
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [7, 6, 6, 7], [6, 7, 7, 6], [5, 7, 7, 7]
    ]
    for pat in patterns:
        for r_est in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.02, 0.98)
            inits.append(c)
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        f = (0.22 - d) * 0.04
                        forces[i] += d_vec / d * f
                        forces[j] -= d_vec / d * f
            c += forces * 0.01
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

def simulated_annealing(c0, rng):
    c = c0.copy()
    _, best_s, _ = solve_lp_and_grad(c)
    best_c = c.copy()
    
    T = 0.015
    for step in range(2500):
        i = rng.integers(N)
        step_size = 0.008 * (0.998 ** step)
        c_try = c.copy()
        c_try[i] += rng.normal(0, step_size, 2)
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        delta = s_try - best_s
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c = c_try
            _, s_curr, _ = solve_lp_and_grad(c)
            if s_curr > best_s:
                best_s = s_curr
                best_c = c.copy()
        T *= 0.997
    return best_c, best_s

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.01, 0.99)] * (2 * N)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    inits = generate_inits(rng)
    
    for c0 in inits:
        try:
            res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 3000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    best_c, best_s = simulated_annealing(best_c, rng)
    best_r, _, _ = solve_lp_and_grad(best_c)
    
    try:
        res = minimize(obj_grad, best_c.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds_lbfgs, options={'maxiter': 4000, 'ftol': 1e-13})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
    except Exception:
        pass
        
    c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
    if s_sl > best_s:
        best_c = c_sl
        best_r = r_sl
        best_s = s_sl
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
