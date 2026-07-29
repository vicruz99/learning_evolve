# sol_000330 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=34724795 sum of radii=2.624008 correctness=1.0
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
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
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
    c = np.clip(x_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def optimize_lbfgs(c0):
    """Runs L-BFGS-B optimization from a starting configuration."""
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    try:
        res = minimize(lbfgs_obj_grad, c0.flatten(), method='L-BFGS-B',
                       bounds=bounds, jac=True,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        c_opt = res.x.reshape(N, 2)
        _, s_opt, _ = solve_lp_and_grad(c_opt)
        return c_opt, s_opt
    except Exception:
        return c0, 0.0

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared for smoothness)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def slsqp_polish(c0, r0):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_joint(res.x)) >= -1e-8:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
            [5, 5, 6, 6, 4], [6, 5, 4, 6, 5], [5, 6, 6, 4, 5],
            [5, 4, 6, 5, 6], [4, 6, 5, 6, 5], [7, 6, 6, 7], [6, 7, 7, 6]]
    
    for pat in pats:
        for r0 in [0.092, 0.096, 0.100, 0.104, 0.108]:
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
                    if d < 0.24 and d > 1e-4:
                        push = (0.24 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-12:
                radii[i] = max(mr, 0.0)
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
        c_opt, s_opt = optimize_lbfgs(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        _, best_s, _ = solve_lp_and_grad(best_c)
        
    best_r, _, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Basin Hopping with pair swaps & perturbations
    curr_c = best_c.copy()
    curr_s = best_s
    step = 0.018
    for k in range(120):
        # Perturb a subset of circles
        n_pert = rng.integers(3, 8)
        idx = rng.choice(N, size=n_pert, replace=False)
        c_try = curr_c.copy()
        c_try[idx] += rng.normal(0, step, (n_pert, 2))
        
        # Occasionally swap two random circles to break symmetry
        if k % 10 == 0:
            s_idx = rng.choice(N, 2, replace=False)
            c_try[s_idx[0]], c_try[s_idx[1]] = c_try[s_idx[1]], c_try[s_idx[0]]
            
        c_try = np.clip(c_try, 0.02, 0.98)
        c_opt, s_opt = optimize_lbfgs(c_try)
        
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            curr_c = best_c.copy()
            curr_s = best_s
            best_r, _, _ = solve_lp_and_grad(best_c)
        elif rng.random() < np.exp((s_opt - curr_s) / max(step * 2, 1e-9)):
            curr_c = c_opt
            curr_s = s_opt
            
        step *= 0.985
        
    # Phase 3: Micro-perturbation refinement (coordinate-wise)
    for _ in range(40):
        c_micro = best_c.copy()
        changed_micro = False
        for i in range(N):
            for d in range(4):
                shift = 0.0015 * ([1, -1, 0, 0][d] if d < 2 else [0, 0, 1, -1][d])
                c_try = c_micro.copy()
                c_try[i] += shift
                c_try = np.clip(c_try, 0.01, 0.99)
                _, s_try, _ = solve_lp_and_grad(c_try)
                if s_try > best_s:
                    c_micro[i] = c_try[i]
                    best_s = s_try
                    changed_micro = True
        if changed_micro:
            best_c = c_micro.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            # Re-optimize fully after micro changes
            c_opt, s_opt = optimize_lbfgs(best_c)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt
                best_r, _, _ = solve_lp_and_grad(best_c)

    # Phase 4: SLSQP Joint Polish for final precision
    c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl
        best_r = r_sl
        
    # Final exact LP solve to guarantee radii match centers
    final_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_s:
        best_r = final_r
        best_s = final_s
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
