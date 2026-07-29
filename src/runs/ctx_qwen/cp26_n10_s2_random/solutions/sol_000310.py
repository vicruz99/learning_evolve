# sol_000310 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=8e4569a6 sum of radii=2.601693 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def setup_lp():
    """Precompute constant LP constraint matrix structure."""
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A = np.zeros((num_pairs + num_bound, N))
    pairs = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[k, i] = 1.0
            A[k, j] = 1.0
            pairs.append((i, j))
            k += 1
    for i in range(N):
        for _ in range(4):
            A[k, i] = 1.0
            k += 1
    return A, pairs

A_LP, PAIR_IDX = setup_lp()

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact gradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-20)
    
    b = np.zeros(A_LP.shape[0])
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
                  bounds=[(0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except Exception:
            duals = np.zeros(len(b))
            
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-8:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    bound_start = len(PAIR_IDX)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return radii, np.sum(radii), grad

def lbfgs_wrapper(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x_flat.reshape(N, 2), 1e-4, 1.0 - 1e-4)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def generate_starts(rng):
    """Generates a diverse set of initial configurations."""
    starts = []
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [7, 6, 6, 7], [6, 7, 7, 6], [5, 7, 7, 7]
    ]
    for pat in pats:
        for r_est in [0.092, 0.098, 0.105, 0.112]:
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
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            forces = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            dists = np.maximum(dists, 1e-5)
            rep = 0.02 / (dists**2)
            for d in range(2):
                forces[:, d] = np.sum(diff[:, :, d] * rep, axis=1)
            c += forces * 0.005
            c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    return starts

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Constraints for SLSQP: boundaries and non-overlap (squared for smoothness)."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
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
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-11:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lb = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B Center Optimization
    for c0 in starts:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lb, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Adaptive Perturbation to escape local minima
    for step in range(50):
        scale = 0.008 * (0.90 ** (step // 5))
        c_trial = best_c + rng.normal(0, scale, best_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        try:
            res = minimize(lbfgs_wrapper, c_trial.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lb, options={'maxiter': 1500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Joint SLSQP Polish for high-precision refinement
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    for _ in range(6):
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl,
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
            
    # Phase 4: Final LP Verification & Strict Numerical Repair
    lp_r, lp_s, _ = solve_lp_and_grad(best_c)
    if lp_s > best_sum:
        best_r = lp_r
        best_sum = lp_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
