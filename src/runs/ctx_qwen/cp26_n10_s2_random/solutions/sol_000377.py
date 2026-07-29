# sol_000377 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000303 (state 682ce44f) state=e152847c sum of radii=2.324941 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N
TRIU_I, TRIU_J = np.triu_indices(N, 1)

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
    A_LP[idx + 4*i, i] = 1.0
    A_LP[idx + 4*i + 1, i] = 1.0
    A_LP[idx + 4*i + 2, i] = 1.0
    A_LP[idx + 4*i + 3, i] = 1.0

# Precomputed bounds for optimizers
BOUNDS_LBFGS = [(0.01, 0.99)] * (2 * N)
BOUNDS_SLSQP = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii and computes exact gradient w.r.t centers using duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
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
        
        active_mask = duals[:NUM_PAIRS] > 1e-9
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

def lbfgs_obj_grad(x):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for SLSQP: boundaries and non-overlap (squared for smoothness)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
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
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Hexagonal lattice patterns with various row distributions
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4], [6,6,5,5,4], 
            [5,5,5,5,6], [7,6,6,7], [6,7,7,6], [5,4,6,6,5], [4,5,6,5,6]]
    for pat in pats:
        for r_est in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    # 2. Corner/Edge biased starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        edges = [[0.5, 0.06], [0.5, 0.94], [0.06, 0.5], [0.94, 0.5]]
        fixed = corners + edges
        c[:8] = fixed
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(c)
        
    # 3. Force-directed spreads
    for _ in range(8):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            diff = c[:, None, :] - c[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            push = np.clip((0.20 - d), 0, None) / (d + 1e-5)
            f = np.sum(push[:, :, None] * diff / (d[:, :, None] + 1e-5), axis=1)
            c += f * 0.01
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)

    # Phase 1: Multi-start L-BFGS-B Center Optimization
    for c0 in inits:
        try:
            res = minimize(lbfgs_obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=BOUNDS_LBFGS, options={'maxiter': 4000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Adaptive Simulated Annealing to escape local minima
    c_curr = best_c.copy()
    s_curr = best_sum
    T = 0.008
    for step in range(1200):
        scale = 0.005 * np.sqrt(T)
        c_try = c_curr + rng.normal(0, scale, c_curr.shape)
        c_try = np.clip(c_try, 0.01, 0.99)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c_curr = c_try
            s_curr = s_try
            if s_curr > best_sum:
                best_sum = s_curr
                best_c = c_curr.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.995
        
    # Phase 3: L-BFGS-B polish on SA result
    try:
        res = minimize(lbfgs_obj_grad, best_c.flatten(), method='L-BFGS-B', jac=True,
                       bounds=BOUNDS_LBFGS, options={'maxiter': 3000, 'ftol': 1e-13})
        c_opt = res.x.reshape(N, 2)
        r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
    except Exception:
        pass
        
    # Phase 4: SLSQP Joint Polish for high-precision center-radius tuning
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(3):
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=BOUNDS_SLSQP,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if np.min(slsqp_cons(res.x)) >= -1e-7:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 5: Final LP Verification & Strict Repair
    lp_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_sum:
        best_r = lp_r
        best_sum = final_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
