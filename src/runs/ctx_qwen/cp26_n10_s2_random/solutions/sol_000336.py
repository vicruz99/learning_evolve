# sol_000336 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=9959f875 sum of radii=2.336306 correctness=1.0
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
    """Solves LP for maximal radii and computes exact subgradient via duals."""
    c = centers
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-24)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = c[i, 0]; k += 1
        b[k] = 1.0 - c[i, 0]; k += 1
        b[k] = c[i, 1]; k += 1
        b[k] = 1.0 - c[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(c)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(c)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def lbfgs_obj_grad(v):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = v.reshape(N, 2)
    c = np.clip(c, 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def opt_lbfgs(c0, max_iter=5000):
    """Optimizes centers using L-BFGS-B with exact LP gradient."""
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    try:
        res = minimize(lbfgs_obj_grad, c0.flatten(), jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5]]
    
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110, 0.115]:
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
            
    for _ in range(30):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.28 and d > 1e-4:
                        push = (0.28 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    for _ in range(20):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint SLSQP: boundary and pairwise non-overlap (squared)."""
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

def slsqp_polish(c0, r0, max_iter=15000):
    """Refines centers and radii jointly using SLSQP."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        c_val = cons_joint(res.x)
        if np.min(c_val) >= -1e-6:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(300):
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
    
    # Phase 1: Multi-start L-BFGS-B
    for c_init in starts:
        c_opt, s_opt = opt_lbfgs(c_init, 5000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, _, _ = solve_lp_and_grad(best_c)
        
        # Phase 2: Targeted Perturbation & Refinement
        for step in range(80):
            noise = 0.015 * (0.90 ** (step // 10))
            c_k = best_c.copy()
            idx = rng.choice(N, size=N, replace=True)
            c_k[idx] += rng.normal(0, noise, (N, 2))
            c_k = np.clip(c_k, 0.02, 0.98)
            
            c_kk, s_kk = opt_lbfgs(c_k, 3000)
            if s_kk > best_s:
                best_s = s_kk
                best_c = c_kk.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
                
        # Phase 3: Simulated Annealing to escape local minima
        c_bh = best_c.copy()
        s_bh = best_s
        T = 0.008
        for step in range(1000):
            c_try = c_bh + rng.normal(0, 0.004, c_bh.shape)
            c_try = np.clip(c_try, 0.02, 0.98)
            _, s_try, _ = solve_lp_and_grad(c_try)
            
            if s_try > s_bh or np.exp((s_try - s_bh) / max(T, 1e-9)) > rng.random():
                c_bh, s_bh = c_try, s_try
                if s_bh > best_s:
                    best_s = s_bh
                    best_c = c_bh.copy()
                    best_r, _, _ = solve_lp_and_grad(best_c)
            T *= 0.993
            
        # Phase 4: SLSQP Joint Polish for final precision
        c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
        if s_sl > best_s:
            best_c = c_sl
            best_r = r_sl
            best_s = s_sl
            
    # Phase 5: Strict numerical repair to guarantee validation passes
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
