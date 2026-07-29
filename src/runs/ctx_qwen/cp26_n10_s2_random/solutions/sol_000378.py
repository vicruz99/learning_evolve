# sol_000378 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000303 (state 682ce44f) state=cfbb9e88 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

# Precompute LP constraint matrix structure globally
A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_I = np.zeros(NUM_PAIRS, dtype=int)
PAIR_J = np.zeros(NUM_PAIRS, dtype=int)
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_I[idx] = i
        PAIR_J[idx] = j
        idx += 1

for i in range(N):
    A_LP[NUM_PAIRS + 4*i, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 1, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 2, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii and computes exact gradient w.r.t centers using duals."""
    n = centers.shape[0]
    
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(NUM_PAIRS + NUM_BOUND)
    b_ub[:NUM_PAIRS] = dists[np.triu_indices(n, 1)]
    
    for i in range(n):
        b_ub[NUM_PAIRS + 4*i] = centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 1] = 1.0 - centers[i, 0]
        b_ub[NUM_PAIRS + 4*i + 2] = centers[i, 1]
        b_ub[NUM_PAIRS + 4*i + 3] = 1.0 - centers[i, 1]
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
            
        radii = res.x
        
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            try:
                duals = np.asarray(res.ineqlin.marginals)
            except Exception:
                duals = np.zeros(len(b_ub))
                
        grad = np.zeros_like(centers)
        
        active_mask = duals[:NUM_PAIRS] > 1e-8
        if np.any(active_mask):
            i_idx = PAIR_I[active_mask]
            j_idx = PAIR_J[active_mask]
            d = dists[np.ix_(i_idx, j_idx)].flatten()
            safe_d = np.where(d > 1e-9, d, 1e-9)
            vec = (centers[i_idx] - centers[j_idx]) / safe_d[:, np.newaxis]
            lam = duals[:NUM_PAIRS][active_mask][:, np.newaxis]
            grad[i_idx] += vec * lam
            grad[j_idx] -= vec * lam
            
        idx_base = NUM_PAIRS + 4 * np.arange(n)
        grad[:, 0] += duals[idx_base] - duals[idx_base + 1]
        grad[:, 1] += duals[idx_base + 2] - duals[idx_base + 3]
            
        return radii, np.sum(radii), grad
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)

def lbfgs_wrapper(x):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    inits = []
    
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [6, 6, 6, 4, 4], [7, 6, 6, 7], [6, 7, 7, 6]
    ]
    
    for pat in patterns:
        if sum(pat) != N:
            continue
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += np.sqrt(3.0) * r0
            inits.append(np.clip(np.array(c[:N]), 0.02, 0.98))
            
    # Force-directed random spreads
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            forces = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            dists = np.maximum(dists, 1e-7)
            rep = 0.015 / (dists**2)
            forces[:, 0] = np.sum(diff[:, :, 0] * rep, axis=1)
            forces[:, 1] = np.sum(diff[:, :, 1] * rep, axis=1)
            c += forces * 0.004
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # Corner/Edge biased
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    return inits

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for SLSQP: boundaries and non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_c = [(0.01, 0.99)] * (2 * N)
    bounds_j = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c, best_r, best_s = None, None, -1.0
    
    # Phase 1: Multi-start L-BFGS-B Center Optimization
    inits = generate_starts(rng)
    for c0 in inits:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 5000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Coordinate Descent Refinement
    curr_c = best_c.copy()
    curr_r, curr_s, _ = solve_lp_and_grad(curr_c)
    for _ in range(3):
        for i in range(N):
            def obj_1d(xy):
                temp = curr_c.copy()
                temp[i] = np.clip(xy, 0.01, 0.99)
                _, s, _ = solve_lp_and_grad(temp)
                return -s
            try:
                res = minimize(obj_1d, curr_c[i], method='L-BFGS-B',
                               bounds=[(0.01, 0.99), (0.01, 0.99)],
                               options={'maxiter': 600, 'ftol': 1e-13})
                if -res.fun > curr_s + 1e-8:
                    curr_c[i] = res.x
                    curr_r, curr_s, _ = solve_lp_and_grad(curr_c)
            except Exception:
                pass
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            
    # Phase 3: Basin-Hopping with subset perturbation
    c_bh = best_c.copy()
    s_bh = best_s
    T = 0.018
    for step in range(250):
        idx = rng.choice(N, size=max(4, N // 3), replace=False)
        c_try = c_bh.copy()
        c_try[idx] += rng.normal(0, 0.012 * np.sqrt(T), (len(idx), 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        try:
            res = minimize(lbfgs_wrapper, c_try.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
        except Exception:
            s_opt = s_bh
            
        delta = s_opt - s_bh
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c_bh = c_opt
            s_bh = s_opt
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.985
        
    # Phase 4: Joint SLSQP Polish for high-precision refinement
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(6):
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_j,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if np.min(slsqp_cons(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_s:
                    best_s = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 5: Final LP Verification & Strict Repair
    lp_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_s:
        best_r = lp_r
        best_s = final_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
