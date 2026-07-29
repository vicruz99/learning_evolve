# sol_000344 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=37d9ed17 sum of radii=2.630972 correctness=1.0
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
    """Solves LP for maximal radii given fixed centers and computes gradient via duals."""
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
        try:
            duals = res.marginals.ineqlin
        except AttributeError:
            duals = np.zeros(len(b))
    else:
        duals = np.zeros(len(b))
        
    radii = res.x if res.success else np.zeros(N)
    s_sum = np.sum(radii)
    
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
        
    return radii, s_sum, grad, duals

def gradient_ascent(centers0, max_iter=4000, init_step=0.006):
    """Gradient ascent on centers using LP dual subgradients."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = init_step
    momentum = np.zeros_like(centers)
    
    _, best_s, _, _ = solve_lp_and_grad(centers)
    
    for _ in range(max_iter):
        r, s, g, _ = solve_lp_and_grad(centers)
        if s > best_s:
            best_s = s
            best_c = centers.copy()
            
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
            
        g_dir = g / gn
        momentum = 0.4 * momentum + step * g_dir
        
        nc = centers + momentum
        nc = np.clip(nc, 1e-5, 1.0 - 1e-5)
        
        _, ns, _, _ = solve_lp_and_grad(nc)
        
        if ns > s + 1e-12:
            centers = nc
            step = min(step * 1.04, 0.045)
            if ns > best_s:
                best_s = ns
                best_c = centers.copy()
        else:
            step *= 0.65
            momentum *= 0.25
            if step < 1e-11:
                break
                
    return best_c, best_s

def force_repel(centers, steps=600, strength=0.012):
    """Spreads circles apart using repulsive forces."""
    c = centers.copy()
    for _ in range(steps):
        diffs = c[:, None, :] - c[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        dists = np.maximum(dists, 1e-6)
        inv_d = 1.0 / dists
        np.fill_diagonal(inv_d, 0.0)
        
        force_mag = np.where(dists < 0.28, strength * inv_d**2, 0.0)
        f = np.zeros_like(c)
        for d in range(2):
            f[:, d] = np.sum(diffs[:, :, d] * force_mag / dists, axis=1)
            
        # Gentle boundary repulsion to keep circles well inside
        mask_low = c < 0.04
        mask_high = c > 0.96
        f[mask_low] += strength * 8.0
        f[mask_high] -= strength * 8.0
        
        c += f * 0.004
        c = np.clip(c, 0.01, 0.99)
    return c

def slsqp_obj(v):
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def slsqp_polish(centers, radii):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return centers, radii, np.sum(radii)

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
    # Hexagonal patterns with varying densities
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5], [5, 5, 5, 5, 6]]
    for pat in pats:
        for r0 in [0.094, 0.099, 0.104, 0.109]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Force-directed layouts
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c = force_repel(c, steps=700, strength=0.015)
        starts.append(c)
        
    # Corner-biased starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))

    # Phase 1: Multi-start Gradient Ascent + SLSQP Polish
    for c_init in starts:
        c_ga, s_ga = gradient_ascent(c_init, 3500, 0.007)
        r_ga, _, _, _ = solve_lp_and_grad(c_ga)
        
        c_sl, r_sl, s_sl = slsqp_polish(c_ga, r_ga)
        
        curr_c, curr_r, curr_s = c_sl, r_sl, s_sl
        if curr_s < s_ga:
            curr_c, curr_r, curr_s = c_ga, r_ga, s_ga
            
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            
    # Phase 2: Simulated Annealing with Perturbations
    T = 0.009
    for step in range(800):
        c_try = best_c.copy()
        
        # Randomized perturbation strategy
        strategy = rng.integers(0, 3)
        if strategy == 0:
            # Kick subset of circles
            idx = rng.choice(N, size=rng.integers(2, 6), replace=False)
            scale = 0.018 * (1.0 - step / 900.0)
            c_try[idx] += rng.normal(0, scale, (len(idx), 2))
        elif strategy == 1:
            # Swap two circles
            i, j = rng.choice(N, 2, replace=False)
            c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
        else:
            # Global small jitter
            c_try += rng.normal(0, 0.004 * (1.0 - step / 900.0), c_try.shape)
            
        c_try = np.clip(c_try, 0.02, 0.98)
        
        c_opt, s_opt = gradient_ascent(c_try, 1800, 0.005)
        
        delta = s_opt - best_s
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            best_c = c_opt.copy()
            best_s = s_opt
            best_r, _, _, _ = solve_lp_and_grad(best_c)
        T *= 0.994
        
    # Phase 3: Shrink-Push-Expand Cycles to escape topological traps
    for cycle in range(6):
        shrink_factor = 0.82 + cycle * 0.015
        r_shrink = best_r * shrink_factor
        c_push = force_repel(best_c, steps=400, strength=0.018)
        c_opt, s_opt = gradient_ascent(c_push, 2500, 0.006)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 4: Final SLSQP Polish on best configuration
    c_final, r_final, s_final = slsqp_polish(best_c, best_r)
    if s_final > best_s:
        best_s = s_final
        best_c = c_final
        best_r = r_final
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
