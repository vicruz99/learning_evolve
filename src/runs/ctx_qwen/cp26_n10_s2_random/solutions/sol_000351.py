# sol_000351 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000344 (state 37d9ed17) state=57458d0f sum of radii=2.624513 correctness=1.0
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
    
    radii = np.zeros(N)
    duals = np.zeros(len(b))
    s_sum = 0.0
    grad = np.zeros_like(centers)
    
    if res.success:
        radii = res.x
        s_sum = np.sum(radii)
        try:
            duals = res.marginals.ineqlin
        except AttributeError:
            try:
                duals = res.ineqlin.marginals
            except AttributeError:
                pass
            
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
            
    return radii, s_sum, grad

def lbfgs_wrapper(x_flat):
    """Objective and exact gradient for L-BFGS-B optimization of centers."""
    c = x_flat.reshape(N, 2)
    r, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4], [6,6,5,5,4], [5,5,5,5,6]]
    for pat in pats:
        for r0 in [0.094, 0.099, 0.104, 0.109, 0.114]:
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
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Force-directed spreads
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            diffs = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            dists = np.maximum(dists, 1e-6)
            mask = dists < 0.28
            inv_d = np.where(mask, 1.0/dists, 0.0)
            np.fill_diagonal(inv_d, 0.0)
            f = np.sum(diffs * inv_d[:, :, None] / dists[:, :, None], axis=1) * 0.006
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Corner-biased starts
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))
        
    return starts

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap."""
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
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
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
    
    bounds_c = [(0.01, 0.99)] * (2 * N)
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B Optimization
    for c0 in starts:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-10})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Basin-Hopping / Kick & Optimize
    for step in range(200):
        scale = 0.018 * (1.0 - step / 220.0)
        c_try = best_c.copy()
        
        strategy = rng.integers(0, 3)
        if strategy == 0:
            idx = rng.choice(N, size=rng.integers(2, 7), replace=False)
            c_try[idx] += rng.normal(0, scale, (len(idx), 2))
        elif strategy == 1:
            i, j = rng.choice(N, 2, replace=False)
            c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
        else:
            c_try += rng.normal(0, scale * 0.5, c_try.shape)
            
        c_try = np.clip(c_try, 0.02, 0.98)
        
        try:
            res = minimize(lbfgs_wrapper, c_try.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
            elif rng.random() < np.exp((s_opt - best_s) / 0.004):
                best_c = c_opt.copy()
                best_r = r_opt.copy()
                best_s = s_opt
        except Exception:
            pass

    # Phase 3: Shrink-Push-Expand Cycles to escape topological traps
    for cycle in range(6):
        factor = 0.82 + cycle * 0.015
        c_push = best_c.copy()
        
        # Repulsive force spread
        for _ in range(400):
            diffs = c_push[:, None, :] - c_push[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            dists = np.maximum(dists, 1e-6)
            mask = dists < 0.30
            inv_d = np.where(mask, 1.0/dists, 0.0)
            np.fill_diagonal(inv_d, 0.0)
            f = np.sum(diffs * inv_d[:, :, None] / dists[:, :, None], axis=1) * 0.009
            c_push += f
            c_push = np.clip(c_push, 0.02, 0.98)
            
        try:
            res = minimize(lbfgs_wrapper, c_push.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass

    # Phase 4: Final SLSQP Polish
    c_final, r_final, s_final = slsqp_polish(best_c, best_r)
    if s_final > best_s:
        best_s = s_final
        best_c = c_final
        best_r = r_final
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
