# sol_000376 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000303 (state 682ce44f) state=96e41a66 sum of radii=2.337910 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N

# Precompute LP structure globally for speed
A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
PAIR_I = np.empty(NUM_PAIRS, dtype=int)
PAIR_J = np.empty(NUM_PAIRS, dtype=int)
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        PAIR_I[k] = i
        PAIR_J[k] = j
        k += 1
for i in range(N):
    A_LP[NUM_PAIRS + 4*i, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 1, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 2, i] = 1.0
    A_LP[NUM_PAIRS + 4*i + 3, i] = 1.0

# Precompute triangular indices for SLSQP constraints
TRIU_I, TRIU_J = np.triu_indices(N, 1)

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii and computes exact gradient w.r.t centers using duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.empty(NUM_PAIRS + NUM_BOUND)
    b_ub[:NUM_PAIRS] = dists[TRIU_I, TRIU_J]
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
        duals = np.zeros(NUM_PAIRS + NUM_BOUND)
        if hasattr(res, 'marginals') and res.marginals is not None:
            duals = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
            duals = np.asarray(res.ineqlin.marginals)
            
        grad = np.zeros_like(centers)
        mask = duals[:NUM_PAIRS] > 1e-9
        if np.any(mask):
            i_idx = PAIR_I[mask]
            j_idx = PAIR_J[mask]
            d = np.hypot(centers[i_idx, 0] - centers[j_idx, 0], 
                         centers[i_idx, 1] - centers[j_idx, 1])
            safe_d = np.where(d > 1e-9, d, 1e-9)
            vec = (centers[i_idx] - centers[j_idx]) / safe_d[:, np.newaxis]
            lam = duals[:NUM_PAIRS][mask][:, np.newaxis]
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
    c = np.clip(x.reshape(N, 2), 0.001, 0.999)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for SLSQP: boundaries and non-overlap (squared distances)."""
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
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-10:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [5, 6, 4, 5, 6], [6, 4, 5, 6, 5], [5, 5, 4, 6, 6]
    ]
    for pat in patterns:
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
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.05, 0.95)
            inits.append(c)
            
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        inits.append(c)
        
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c[:4] = corners
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.001, 0.999)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c0 in inits:
        try:
            res = minimize(lbfgs_obj_grad, c0.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-13})
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
        
    # Phase 2: Iterated Local Search (ILS)
    for step in range(60):
        scale = 0.015 * (0.90 ** (step // 5))
        c_trial = best_c.copy()
        k = rng.integers(1, 6)
        idx = rng.choice(N, k, replace=False)
        c_trial[idx] += rng.normal(0, scale, (k, 2))
        c_trial = np.clip(c_trial, 0.01, 0.99)
        
        try:
            res = minimize(lbfgs_obj_grad, c_trial.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_lbfgs, options={'maxiter': 1500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Topology Swap Strategy
    for _ in range(50):
        c_swap = best_c.copy()
        i, j = rng.choice(N, 2, replace=False)
        c_swap[i], c_swap[j] = c_swap[j], c_swap[i]
        c_swap[i] += rng.normal(0, 0.005, 2)
        c_swap[j] += rng.normal(0, 0.005, 2)
        c_swap = np.clip(c_swap, 0.01, 0.99)
        
        try:
            res = minimize(lbfgs_obj_grad, c_swap.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_lbfgs, options={'maxiter': 1200, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 4: Fine-tune single circles
    for step in range(40):
        c_fine = best_c.copy()
        idx = rng.integers(N)
        c_fine[idx] += rng.normal(0, 0.002, 2)
        c_fine = np.clip(c_fine, 0.01, 0.99)
        
        try:
            res = minimize(lbfgs_obj_grad, c_fine.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_lbfgs, options={'maxiter': 800, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 5: SLSQP Joint Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(4):
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': slsqp_cons},
                          options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(slsqp_cons(res.x)) >= -1e-7:
                s = np.sum(res.x[2*N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2*N].reshape(N, 2).copy()
                    best_r = res.x[2*N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 6: Final LP Verification & Strict Repair
    lp_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_sum:
        best_r = lp_r
        best_sum = final_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
