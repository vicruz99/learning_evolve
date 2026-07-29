# sol_000322 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000291 (state e7ab877e) state=791f23d5 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

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

def val_lp(x):
    """Objective wrapper for L-BFGS-B."""
    c = x.reshape(N, 2)
    _, s, _ = solve_lp_and_grad(c)
    return -s

def jac_lp(x):
    """Gradient wrapper for L-BFGS-B."""
    c = x.reshape(N, 2)
    _, _, g = solve_lp_and_grad(c)
    return -g.flatten()

def constraints_joint(v):
    """Computes boundary and non-overlap constraints for joint SLSQP optimization."""
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

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def hex_init(r_est, rng, pattern=None):
    """Generates hexagonal lattice initialization."""
    if pattern is None:
        pattern = [5, 6, 5, 6, 4]
    c = []
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
    c = np.array(c[:N])
    c += rng.normal(0, 0.003, c.shape)
    return np.clip(c, 0.02, 0.98)

def force_init(rng):
    """Generates a well-spaced configuration via repulsive force growth."""
    c = rng.uniform(0.2, 0.8, (N, 2))
    r = np.full(N, 0.01)
    for _ in range(1500):
        r += 0.0003
        diffs = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, 1e9)
        overlap = np.maximum(r[:, None] + r[None, :] - dists, 0)
        safe_d = np.where(dists > 1e-8, dists, 1e-8)
        f = np.zeros_like(c)
        mask = dists > 1e-8
        dirs = np.where(mask[:, :, None], diffs / safe_d[:, :, None], 0)
        mag = overlap / (safe_d**2 + 1e-6)
        f = np.sum(mag[:, :, None] * dirs, axis=1)
        c += 0.02 * f
        c = np.clip(c, 0.0, 1.0)
    return c

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(80):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = []
    patterns = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [6,6,5,5,4], [4,6,6,6,4]]
    for pat in patterns:
        for r0 in [0.092, 0.098, 0.105]:
            starts.append(hex_init(r0, rng, pat))
            
    for _ in range(10):
        starts.append(force_init(rng))
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_c = [(0.001, 0.999)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    candidates = []
    
    # Phase 1: L-BFGS-B on all starts using exact LP gradients
    for c_init in starts:
        try:
            res = minimize(val_lp, c_init.flatten(), jac=jac_lp, method='L-BFGS-B',
                           bounds=bounds_c, options={'maxiter': 1500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            candidates.append((s_opt, c_opt, r_opt))
        except Exception:
            pass
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Targeted Perturbation on top candidates to escape local minima
    for s, c, r in candidates[:5]:
        for _ in range(8):
            c_p = c + rng.normal(0, 0.005, c.shape)
            c_p = np.clip(c_p, 0.02, 0.98)
            try:
                res = minimize(val_lp, c_p.flatten(), jac=jac_lp, method='L-BFGS-B',
                               bounds=bounds_c, options={'maxiter': 1000, 'ftol': 1e-13})
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
                if s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
            except Exception:
                pass
                
    if best_c is None and candidates:
        best_s, best_c, best_r = candidates[0]
        
    # Phase 3: SLSQP Joint Polish for high-precision center-radius tuning
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        try:
            res = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_joint,
                           constraints={'type': 'ineq', 'fun': constraints_joint},
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraints_joint(res.x)) >= -1e-7:
                s_sl = np.sum(res.x[2*N:])
                if s_sl > best_s:
                    best_s = s_sl
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
        except Exception:
            pass
            
    # Phase 4: Simulated Annealing for final refinement
    if best_c is not None:
        c_curr = best_c.copy()
        s_curr = best_s
        T = 0.006
        for step in range(400):
            c_try = c_curr + rng.normal(0, 0.004, c_curr.shape)
            c_try = np.clip(c_try, 0.02, 0.98)
            _, s_try, _ = solve_lp_and_grad(c_try)
            delta = s_try - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                c_curr, s_curr = c_try, s_try
                if s_curr > best_s:
                    best_s = s_curr
                    best_c = c_curr.copy()
                    best_r, _, _ = solve_lp_and_grad(best_c)
            T *= 0.99
            
    # Final exact LP solve to sync radii with optimized centers
    if best_c is not None:
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Strict numerical repair to guarantee validator passes within 1e-12 tolerance
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
