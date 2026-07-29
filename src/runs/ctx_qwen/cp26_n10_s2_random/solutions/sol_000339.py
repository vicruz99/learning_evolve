# sol_000339 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=fe4e0695 sum of radii=2.609582 correctness=1.0
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

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
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
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
        s = np.sum(radii)
        try:
            duals = res.marginals.ineqlin
        except AttributeError:
            duals = np.zeros(len(b))
            
        grad = np.zeros_like(centers)
        k = 0
        for i, j in PAIR_IDX:
            mu = duals[k]
            if mu > 1e-8:
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
            
        return radii, s, grad
    return np.zeros(N), 0.0, np.zeros_like(centers)

def obj_func(v):
    """Objective for L-BFGS-B: minimizes negative sum of radii."""
    c = v.reshape(N, 2)
    _, s, _ = solve_lp_and_grad(c)
    return -s

def jac_func(v):
    """Gradient for L-BFGS-B."""
    c = v.reshape(N, 2)
    _, _, g = solve_lp_and_grad(c)
    return -g.flatten()

def optimize_centers_lbgbs(c0, max_iter=4000):
    """Optimizes centers using L-BFGS-B."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(obj_func, c0.flatten(), jac=jac_func, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

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
                    push = (0.22 - d) * 0.05
                    f[i] += d_vec / d * push
                    f[j] -= d_vec / d * push
        c += f
        c = np.clip(c, 0.05, 0.95)
    return c

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint(v):
    """Constraints for joint SLSQP: boundary and non-overlap."""
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
    
    starts = []
    
    # Hexagonal patterns with various row distributions
    pats = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5],
        [5, 5, 5, 6, 5], [6, 5, 5, 5, 5], [5, 6, 4, 6, 5],
        [4, 7, 6, 6, 3], [5, 7, 5, 6, 3], [6, 6, 6, 4, 4]
    ]
    for pat in pats:
        for r0 in [0.092, 0.096, 0.100, 0.105]:
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
    for _ in range(10):
        starts.append(force_directed_init(rng))
        
    # Corner-biased starts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))

    # Phase 1: L-BFGS-B optimization from diverse starts
    for c_init in starts:
        c_opt, s_opt = optimize_centers_lbgbs(c_init, max_iter=4000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 2: Basin Hopping / Perturbation refinement
    for step in range(60):
        scale = 0.018 * (0.92 ** (step // 12))
        c_k = best_c.copy()
        idx = rng.choice(N, size=N, replace=True)
        c_k[idx] += rng.normal(0, scale, (N, 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        
        c_opt, s_opt = optimize_centers_lbgbs(c_k, max_iter=2500)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 3: Swap perturbation to break ordering traps
    for _ in range(30):
        c_swap = best_c.copy()
        i, j = rng.choice(N, 2, replace=False)
        c_swap[[i, j]] = c_swap[[j, i]]
        c_opt, s_opt = optimize_centers_lbgbs(c_swap, max_iter=2500)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 4: Joint SLSQP Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_joint(res.x)) >= -1e-7:
            s_sl = np.sum(res.x[2 * N:])
            if s_sl > best_s:
                best_s = s_sl
                best_c = res.x[:2 * N].reshape(N, 2)
                best_r = res.x[2 * N:]
    except Exception:
        pass

    # Final LP polish to ensure radii are maximally consistent with centers
    r_final, s_final, _ = solve_lp_and_grad(best_c)
    if s_final > best_s:
        best_s = s_final
        best_r = r_final
        
    # Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
