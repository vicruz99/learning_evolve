# sol_000338 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=bef8407a sum of radii=2.621304 correctness=1.0
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
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
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
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros(NUM_PAIRS + 4 * N)
    except Exception:
        return np.zeros(N), 0.0, np.zeros(NUM_PAIRS + 4 * N)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(NUM_PAIRS + 4 * N)
    try:
        if hasattr(res, 'marginals') and res.marginals is not None:
            duals = res.marginals.ineqlin
        elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
            duals = res.ineqlin.marginals
    except Exception:
        pass
        
    return radii, s_sum, duals

def compute_grad(centers, duals):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
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

def grad_ascent(centers0, max_iter=1500, step_init=0.005):
    """Gradient ascent on centers with momentum and adaptive step."""
    c = centers0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = step_init
    momentum = np.zeros_like(c)
    
    _, best_s, _ = solve_lp_and_grad(c)
    
    for _ in range(max_iter):
        r, s, d = solve_lp_and_grad(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            
        g = compute_grad(c, d)
        gn = np.linalg.norm(g)
        if gn < 1e-11:
            break
            
        momentum = 0.3 * momentum + step * (g / (gn + 1e-12))
        nc = c + momentum
        nc = np.clip(nc, 1e-5, 1.0 - 1e-5)
        
        _, ns, _ = solve_lp_and_grad(nc)
        if ns > s + 1e-10:
            c = nc
            step = min(step * 1.02, 0.04)
        else:
            step *= 0.75
            momentum *= 0.2
            if step < 1e-9:
                break
    return best_c, best_s

def simulated_annealing(centers, rng, steps=2500):
    """Simulated annealing to escape local minima."""
    c = centers.copy()
    _, s_curr, _ = solve_lp_and_grad(c)
    best_c = c.copy()
    best_s = s_curr
    T = 0.008
    decay = 0.995
    
    for _ in range(steps):
        i = rng.integers(N)
        c_try = c.copy()
        scale = 0.012 * (T / 0.008)
        c_try[i] += rng.normal(0, scale, 2)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c = c_try
            s_curr = s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c.copy()
        T *= decay
    return best_c, best_s

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
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

def slsqp_joint_polish(c0, r0):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-8:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, 0.0

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
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

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 5, 5, 5, 6]]
    for pat in pats:
        for r0 in [0.095, 0.100, 0.105]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            c_arr = np.array(c[:N])
            c_arr += rng.normal(0, 0.003, c_arr.shape)
            starts.append(np.clip(c_arr, 0.05, 0.95))
            
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.22 and d > 1e-4:
                        push = (0.22 - d) * 0.05
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = corners
        starts.append(c)
        
    return starts

def run_packing() -> tuple:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient ascent from multiple starts
    for c0 in starts:
        c_ga, s_ga = grad_ascent(c0, 2000)
        if s_ga > best_s:
            best_s = s_ga
            best_c = c_ga.copy()
            
    if best_c is not None:
        # Phase 2: Targeted kicks to escape local minima
        for _ in range(25):
            c_kick = best_c.copy()
            idx = rng.choice(N, size=4, replace=False)
            c_kick[idx] += rng.normal(0, 0.02, (4, 2))
            c_kick = np.clip(c_kick, 0.05, 0.95)
            c_opt, s_opt = grad_ascent(c_kick, 1000)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                
        # Phase 3: Simulated Annealing for global exploration
        c_sa, s_sa = simulated_annealing(best_c, rng, steps=3000)
        if s_sa > best_s:
            best_s = s_sa
            best_c = c_sa.copy()
            
        # Phase 4: Joint SLSQP Polish for final precision
        r_lp, _, _ = solve_lp_and_grad(best_c)
        c_sl, r_sl, s_sl = slsqp_joint_polish(best_c, r_lp)
        if s_sl > best_s:
            best_s = s_sl
            best_c = c_sl
            best_r = r_sl
        else:
            best_r = r_lp
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
