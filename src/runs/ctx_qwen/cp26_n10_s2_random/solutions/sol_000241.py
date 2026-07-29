# sol_000241 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000213 (state adb87445) state=5982b738 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
NUM_PAIRS = N * (N - 1) // 2
TRIU_I, TRIU_J = np.triu_indices(N, 1)

# Precompute constant LP constraint matrix structure for speed
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        PAIR_IDX.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and dual multipliers."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    b[:NUM_PAIRS] = dists[TRIU_I, TRIU_J]
    for i in range(n):
        base = NUM_PAIRS + 4 * i
        b[base] = centers[i, 0]
        b[base + 1] = 1.0 - centers[i, 0]
        b[base + 2] = centers[i, 1]
        b[base + 3] = 1.0 - centers[i, 1]
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        # Safe extraction of dual marginals across scipy versions
        duals = np.zeros(len(b))
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            duals = res.ineqlin.marginals
        return res.x, np.sum(res.x), duals
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

def ga_optimize(centers0, max_iter=2500, lr_init=0.006):
    """Gradient ascent on centers to maximize sum of radii with adaptive step size."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    r, s, d = solve_lp(centers)
    best_s = s
    lr = lr_init
    
    for step in range(max_iter):
        g = compute_grad(centers, d)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
            
        move = (lr / gn) * g
        nc = np.clip(centers + move, 1e-5, 1.0 - 1e-5)
        
        nr, ns, nd = solve_lp(nc)
        if ns > best_s:
            best_s = ns
            best_c = nc.copy()
            
        if ns > s + 1e-14:
            centers = nc
            s = ns
            d = nd
            lr = min(lr * 1.08, 0.04)
        else:
            lr *= 0.82
            if lr < 1e-9:
                break
        
        # Periodic kick to escape flat regions
        if step > 0 and step % 150 == 0:
            lr = max(lr, 0.004)
            
    return best_c, best_s

def slsqp_joint(c0, r0):
    """Joint SLSQP optimization of centers and radii for high-precision polishing."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
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
        
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        c_vals = cons(res.x)
        if np.min(c_vals) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, 0.0

def generate_starts(rng):
    """Generates diverse initial configurations including hex lattices and boundary-biased layouts."""
    starts = []
    
    # Hexagonal patterns with varying row counts
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 4, 6, 5, 5], [5, 5, 6, 5, 5], [6, 6, 5, 5, 4]]
    for pat in pats:
        for r0 in [0.092, 0.098, 0.105, 0.110]:
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
            
    # Corner-biased starts to exploit boundary constraints
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        idx = rng.choice(N, 4, replace=False)
        c[idx] = np.array(corners) + rng.normal(0, 0.02, (4, 2))
        starts.append(np.clip(c, 0.02, 0.98))
        
    # Force-directed repulsion layouts
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
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
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(120):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    # Phase 1: Generate and optimize diverse starts
    starts = generate_starts(rng)
    
    for c_init in starts:
        # Quick initial radii guess
        ub = np.minimum(np.minimum(c_init[:, 0], 1.0 - c_init[:, 0]), 
                        np.minimum(c_init[:, 1], 1.0 - c_init[:, 1]))
        dists = np.linalg.norm(c_init[:, None, :] - c_init[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        r_init = np.minimum(ub, 0.5 * np.min(dists, axis=1)) * 0.9
        
        # Gradient Ascent
        c_ga, s_ga = ga_optimize(c_init, max_iter=2000, lr_init=0.007)
        r_ga, _, _ = solve_lp(c_ga)
        
        # SLSQP Polish
        c_sl, r_sl, s_sl = slsqp_joint(c_ga, r_ga)
        
        curr_c, curr_r, curr_s = c_sl, r_sl, s_sl
        if curr_s < s_ga:
            curr_c, curr_r, curr_s = c_ga, r_ga, s_ga
            
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            
    # Phase 2: Iterative Shake & Settle to escape local minima
    c_curr = best_c.copy()
    s_curr = best_s
    r_curr = best_r.copy()
    
    for step in range(60):
        # Perturbation scale decreases over time
        scale = 0.012 * (0.96 ** step)
        c_try = c_curr.copy()
        # Randomly shuffle/perturb a subset of circles
        idx = rng.choice(N, size=max(4, N//3), replace=False)
        c_try[idx] += rng.normal(0, scale, (len(idx), 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        # Fast gradient ascent on perturbed configuration
        c_opt, s_opt = ga_optimize(c_try, max_iter=800, lr_init=0.004)
        
        if s_opt > s_curr:
            c_curr = c_opt
            s_curr = s_opt
            r_curr, _, _ = solve_lp(c_curr)
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r = r_curr.copy()
                
    # Phase 3: Final high-precision joint polish
    c_pol, r_pol, s_pol = slsqp_joint(best_c, best_r)
    if s_pol > best_s:
        best_s = s_pol
        best_c = c_pol
        best_r = r_pol
        
    # Phase 4: Strict numerical repair and final validation check
    r_final, s_final, _ = solve_lp(best_c)
    if s_final > best_s:
        best_s = s_final
        best_r = r_final
        
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
