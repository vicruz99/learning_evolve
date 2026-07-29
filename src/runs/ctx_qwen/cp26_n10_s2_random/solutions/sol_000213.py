# sol_000213 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000168 (state 79899e79) state=adb87445 sum of radii=2.630475 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure
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

def solve_lp(centers):
    """Solves LP for maximal radii given fixed centers."""
    n = centers.shape[0]
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
    for i in range(n):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        return res.x, np.sum(res.x), res.ineqlin.marginals
    return np.zeros(n), 0.0, np.zeros(len(b))

def compute_grad(centers, duals):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
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
    return grad

def ga_optimize(centers0, max_iter=2500):
    """Gradient ascent on centers with adaptive line search."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = 0.006
    
    for _ in range(max_iter):
        r, s, d = solve_lp(centers)
        if s > best_s:
            best_s = s
            best_c = centers.copy()
            
        g = compute_grad(centers, d)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
            
        step = min(step, 0.025)
        move = step * (g / gn)
        nc = np.clip(centers + move, 1e-5, 1.0 - 1e-5)
        _, ns, _ = solve_lp(nc)
        
        if ns > s:
            centers = nc
            step *= 1.15
        else:
            step *= 0.6
            nc2 = np.clip(centers + step * (g / gn), 1e-5, 1.0 - 1e-5)
            _, ns2, _ = solve_lp(nc2)
            if ns2 > s:
                centers = nc2
            else:
                step *= 0.5
                if step < 1e-9:
                    break
    return best_c, best_s

def force_directed_init(rng):
    """Generates a well-spaced configuration via repulsive forces."""
    c = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(600):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = c[i] - c[j]
                d = np.linalg.norm(d_vec)
                if d < 0.22 and d > 1e-4:
                    push = (0.22 - d) * 0.06
                    f[i] += d_vec / d * push
                    f[j] -= d_vec / d * push
        c += f
        c = np.clip(c, 0.05, 0.95)
    return c

def slsqp_joint(c0, r0):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    
    def obj(v):
        return -np.sum(v[2 * N:])

    def cons(v):
        c = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
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
        
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, 0.0

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
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
    
    # Phase 1: Generate diverse initial configurations
    starts = []
    
    # Hexagonal patterns
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]]
    for pat in pats:
        for r0 in [0.092, 0.098, 0.105]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Force-directed layouts
    for _ in range(12):
        starts.append(force_directed_init(rng))
        
    # Random dense starts
    for _ in range(12):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))

    # Phase 2: Multi-start optimization loop
    for c_init in starts:
        # Safe initial radii
        ub = np.minimum(np.minimum(c_init[:, 0], 1.0 - c_init[:, 0]), 
                        np.minimum(c_init[:, 1], 1.0 - c_init[:, 1]))
        dists = np.linalg.norm(c_init[:, None, :] - c_init[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        rp = 0.5 * np.min(dists, axis=1)
        r_init = np.minimum(ub, rp) * 0.88
        
        # Gradient Ascent
        c_ga, s_ga = ga_optimize(c_init, 2000)
        r_ga, _, _ = solve_lp(c_ga)
        
        # SLSQP Joint Polish
        c_sl, r_sl, s_sl = slsqp_joint(c_ga, r_ga)
        
        # Pick best from this start
        curr_c, curr_r, curr_s = c_sl, r_sl, s_sl
        if curr_s < s_ga:
            curr_c, curr_r, curr_s = c_ga, r_ga, s_ga
            
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            
        # Targeted Kicks to escape local minima
        for _ in range(6):
            c_k = best_c.copy()
            idx = rng.choice(N, size=6, replace=False)
            c_k[idx] += rng.normal(0, 0.015, (6, 2))
            c_k = np.clip(c_k, 0.05, 0.95)
            c_kk, s_kk = ga_optimize(c_k, 1200)
            if s_kk > best_s:
                best_s = s_kk
                best_c = c_kk
                r_kk, _, _ = solve_lp(best_c)
                best_r = r_kk
                
    # Phase 3: Basin Hopping Refinement on best configuration
    c_bh = best_c.copy()
    s_bh = best_s
    r_bh = best_r.copy()
    T = 0.008
    for step in range(300):
        c_try = c_bh + rng.normal(0, 0.006, c_bh.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp(c_try)
        
        if s_try > s_bh or np.exp((s_try - s_bh) / T) > rng.random():
            c_bh, s_bh = c_try, s_try
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp(best_c)
        T *= 0.992
        
    # Final LP polish
    r_final, s_final, _ = solve_lp(best_c)
    if s_final > best_s:
        best_s = s_final
        best_r = r_final
        
    # Phase 4: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
