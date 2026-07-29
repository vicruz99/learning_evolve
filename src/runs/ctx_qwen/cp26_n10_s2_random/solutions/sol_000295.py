# sol_000295 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000243 (state e183a9b7) state=12b2463d sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
    return radii, s_sum, grad

def objective_centers(v):
    c = v.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def obj_sl(v):
    return -np.sum(v[2*N:])

def cons_sl(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(np.sqrt(dx**2 + dy**2) - dr)
    return np.concatenate(con)

def slsqp_polish(c0, r0):
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_sl, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_sl},
                       options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
        if np.min(cons_sl(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_inits(rng):
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
            [5, 5, 5, 6, 5], [6, 6, 4, 5, 5], [4, 6, 5, 5, 6]]
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.001), y + rng.normal(0, 0.001)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.02, 0.98))
            
    for _ in range(25):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(300):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.18 and d > 1e-4:
                        push = (0.18 - d) * 0.08 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.02, 0.98)
        starts.append(c)
    return starts

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
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_inits(rng)
    bounds_c = [(0.02, 0.98)] * (2 * N)
    
    # Phase 1: L-BFGS-B on centers
    for c_init in starts:
        v0 = c_init.flatten()
        try:
            res = minimize(objective_centers, v0, method='L-BFGS-B', jac=True, 
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: SLSQP Polish
    if best_c is not None:
        c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
        if s_sl > best_s:
            best_s = s_sl
            best_c = c_sl
            best_r = r_sl
            
    # Phase 3: Kicks + L-BFGS-B + SLSQP
    for _ in range(50):
        c_k = best_c.copy()
        idx = rng.choice(N, size=10, replace=False)
        c_k[idx] += rng.normal(0, 0.015, (10, 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        
        v0 = c_k.flatten()
        try:
            res = minimize(objective_centers, v0, method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
                
                c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
                if s_sl > best_s:
                    best_s = s_sl
                    best_c = c_sl
                    best_r = r_sl
        except Exception:
            pass
            
    # Phase 4: Simulated Annealing
    T = 0.008
    c_curr = best_c.copy()
    s_curr = best_s
    for step in range(1000):
        c_try = c_curr + rng.normal(0, 0.004, c_curr.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        if s_try > s_curr or np.exp((s_try - s_curr) / max(T, 1e-9)) > rng.random():
            c_curr, s_curr = c_try, s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.995
        
    # Final polish
    if best_c is not None:
        c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
        if s_sl > best_s:
            best_c = c_sl
            best_r = r_sl
            best_s = s_sl
            
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
