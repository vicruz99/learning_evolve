# sol_000327 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=915b779b sum of radii=2.624513 correctness=1.0
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

def lbfgs_obj_grad(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x_flat.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def optimize_centers(c0):
    """Optimizes circle positions using L-BFGS-B with exact gradient."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(lbfgs_obj_grad, c0.flatten(), method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    # Hexagonal patterns
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5]]
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105]:
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
            inits.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Force-directed layouts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f * 0.01
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # Random starts
    for _ in range(20):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return inits

def slsqp_obj(v):
    return -np.sum(v[2*N:])

def slsqp_cons(v):
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
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        c_val = slsqp_cons(res.x)
        if np.min(c_val) >= -1e-6:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

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
    
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c_init in inits:
        c_opt, s_opt = optimize_centers(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 2: Basin Hopping / Perturbation to escape local minima
    for it in range(80):
        c_pert = best_c.copy()
        n_pert = rng.integers(3, 10)
        idx = rng.choice(N, size=n_pert, replace=False)
        noise = 0.018 * (0.95 ** (it // 5))
        c_pert[idx] += rng.normal(0, noise, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        c_opt, s_opt = optimize_centers(c_pert)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
        # Swap two circles occasionally to break symmetric traps
        if it % 10 == 0:
            swap_idx = rng.choice(N, 2, replace=False)
            idx_a, idx_b = swap_idx
            best_c[[idx_a, idx_b]] = best_c[[idx_b, idx_a]]
            c_opt, s_opt = optimize_centers(best_c)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)

    # Phase 3: SLSQP Joint Polish for final precision
    c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl
        best_r = r_sl
        
    # Phase 4: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
