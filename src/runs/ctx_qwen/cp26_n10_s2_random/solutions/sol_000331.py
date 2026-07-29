# sol_000331 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=8a8e5e17 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure globally
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
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
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

def ga_optimize(centers0, max_iter=2500, step_init=0.008):
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = step_init
    velocity = np.zeros_like(centers)
    beta = 0.6
    
    for _ in range(max_iter):
        r, s, g = solve_lp_and_grad(centers)
        if s > best_s:
            best_s = s
            best_c = centers.copy()
            
        gn = np.linalg.norm(g)
        if gn < 1e-11:
            break
            
        g_dir = g / gn
        velocity = beta * velocity + step * g_dir
        nc = centers + velocity
        nc = np.clip(nc, 1e-5, 1.0 - 1e-5)
        
        _, ns, _ = solve_lp_and_grad(nc)
        
        if ns > s:
            centers = nc
            step = min(step * 1.08, 0.04)
            velocity *= 1.1
        else:
            step *= 0.75
            velocity *= 0.5
            
        if step < 1e-9:
            break
    return best_c, best_s

def obj_joint(v):
    return -np.sum(v[2*N:])

def cons_joint(v):
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
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def slsqp_polish(c0, r0):
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        c_val = cons_joint(res.x)
        if np.min(c_val) >= -1e-5:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_starts(rng):
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
            [5, 5, 6, 6, 4], [6, 5, 4, 6, 5], [5, 6, 6, 4, 5],
            [5, 4, 6, 5, 6], [4, 6, 5, 6, 5]]
    
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
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.04 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f * 0.5
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def repair(centers, radii):
    radii = radii.copy()
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
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start Gradient Ascent + SLSQP Polish
    for c_init in starts:
        c_ga, s_ga = ga_optimize(c_init, 2500)
        r_ga, _, _ = solve_lp_and_grad(c_ga)
        
        c_sl, r_sl, s_sl = slsqp_polish(c_ga, r_ga)
        
        curr_c, curr_r, curr_s = c_sl, r_sl, s_sl
        if curr_s < s_ga:
            curr_c, curr_r, curr_s = c_ga, r_ga, s_ga
            
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            
    # Phase 2: Perturbation & Symmetry-Breaking Swap Search
    for _ in range(60):
        c_k = best_c.copy()
        n_pert = rng.integers(3, 8)
        idx = rng.choice(N, size=n_pert, replace=False)
        c_k[idx] += rng.normal(0, 0.015, (n_pert, 2))
        
        # Random swap to break symmetric traps
        if rng.random() < 0.3:
            s1, s2 = rng.choice(N, 2, replace=False)
            c_k[[s1, s2]] = c_k[[s2, s1]]
            
        c_k = np.clip(c_k, 0.02, 0.98)
        
        c_kk, s_kk = ga_optimize(c_k, 1500)
        if s_kk > best_s:
            best_s = s_kk
            best_c = c_kk
            r_kk, _, _ = solve_lp_and_grad(best_c)
            best_r = r_kk
            c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
            if s_sl > best_s:
                best_s = s_sl
                best_c = c_sl
                best_r = r_sl
                
    # Phase 3: Simulated Annealing for final basin exploration
    c_sa = best_c.copy()
    s_sa = best_s
    T = 0.006
    for step in range(1000):
        c_try = c_sa + rng.normal(0, 0.005, c_sa.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        if s_try > s_sa or np.exp((s_try - s_sa) / max(T, 1e-9)) > rng.random():
            c_sa, s_sa = c_try, s_try
            if s_sa > best_s:
                best_s = s_sa
                best_c = c_sa.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.994
        
    # Phase 4: Final SLSQP polish for maximum precision
    c_final, r_final, s_final = slsqp_polish(best_c, best_r)
    if s_final > best_s:
        best_c = c_final
        best_r = r_final
        best_s = s_final
        
    # Phase 5: Strict numerical repair to guarantee validation passes
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
