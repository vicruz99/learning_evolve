# sol_000313 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000286 (state d00da21c) state=67befbab sum of radii=2.556932 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute static LP constraint matrix structure
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
    for k in range(4):
        A_LP[NUM_PAIRS + 4 * i + k, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact subgradient."""
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    b_ub = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, None)] * N, method='highs')
        if not res.success:
            return np.full(N, 1e-5), 0.0, np.zeros_like(centers)
    except Exception:
        return np.full(N, 1e-5), 0.0, np.zeros_like(centers)
        
    radii = res.x
    sum_r = np.sum(radii)
    
    duals = np.zeros_like(b_ub)
    try:
        duals = res.marginals.ineqlin
    except AttributeError:
        try:
            duals = res.ineqlin.marginals
        except AttributeError:
            pass
            
    grad_sum_r = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad_sum_r[i] += lam * vec
                grad_sum_r[j] -= lam * vec
        k += 1
        
    for i in range(N):
        mu_L = duals[NUM_PAIRS + 4 * i]
        mu_R = duals[NUM_PAIRS + 4 * i + 1]
        mu_B = duals[NUM_PAIRS + 4 * i + 2]
        mu_T = duals[NUM_PAIRS + 4 * i + 3]
        grad_sum_r[i, 0] += mu_L - mu_R
        grad_sum_r[i, 1] += mu_B - mu_T
        
    # Gradient of objective (-sum_r) for minimization
    return radii, sum_r, -grad_sum_r

def lbfgs_wrapper(v_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(v_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return val, grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 6, 4, 5, 6], [5, 5, 4, 6, 6], [6, 4, 5, 6, 5],
        [7, 6, 6, 7], [6, 7, 7, 6], [5, 7, 7, 7], [8, 5, 8, 5]
    ]
    
    for pat in patterns:
        for r_est in [0.088, 0.093, 0.098, 0.103, 0.108]:
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
            while len(c) < N:
                c.append(rng.uniform(0.2, 0.8, 2))
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            starts.append(np.clip(c, 0.05, 0.95))
            
    # Force-directed spreads
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist = np.maximum(dist, 1e-4)
            rep = np.where(dist < 0.28, 0.025 / (dist**2), 0.0)
            f = np.sum(diff * rep[:, :, None], axis=1)
            c += 0.006 * f
            c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    # Corner-heavy starts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        c += rng.normal(0, 0.01, c.shape)
        starts.append(np.clip(c, 0.05, 0.95))
        
    return starts

def slsqp_obj(v):
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
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
    bounds_lbfgs = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c0 in starts:
        try:
            res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 3000, 'ftol': 1e-14})
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c = np.clip(res.x.reshape(N, 2), 1e-6, 1.0 - 1e-6)
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
    best_r, best_sum, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Iterative Gaussian Perturbation & Local Search
    for step in range(60):
        noise_scale = 0.009 * (0.92 ** (step / 6.0))
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.05, 0.95)
        
        try:
            res = minimize(lbfgs_wrapper, c_pert.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-14})
            c_pert = np.clip(res.x.reshape(N, 2), 1e-6, 1.0 - 1e-6)
            r_pert, s_pert, _ = solve_lp_and_grad(c_pert)
            
            if s_pert > best_sum:
                best_sum = s_pert
                best_c = c_pert.copy()
                best_r = r_pert.copy()
        except Exception:
            continue
            
    # Phase 3: Swap-based topological exploration
    for _ in range(40):
        idx1, idx2 = rng.choice(N, 2, replace=False)
        c_swap = best_c.copy()
        c_swap[[idx1, idx2]] = c_swap[[idx2, idx1]]
        c_swap += rng.normal(0, 0.006, best_c.shape)
        c_swap = np.clip(c_swap, 0.05, 0.95)
        try:
            res = minimize(lbfgs_wrapper, c_swap.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 1500, 'ftol': 1e-14})
            c_swap = np.clip(res.x.reshape(N, 2), 1e-6, 1.0 - 1e-6)
            r_swap, s_swap, _ = solve_lp_and_grad(c_swap)
            if s_swap > best_sum:
                best_sum = s_swap
                best_c = c_swap.copy()
                best_r = r_swap.copy()
        except Exception:
            continue
            
    # Phase 4: Joint SLSQP Polish for high-precision convergence
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(5):
        # Add tiny random jitter to avoid flat gradient regions
        v0_pert = v0 + rng.normal(0, 1e-5, v0.shape)
        v0_pert[:2*N] = np.clip(v0_pert[:2*N], 0.0, 1.0)
        v0_pert[2*N:] = np.maximum(v0_pert[2*N:], 0.0)
        
        try:
            res = minimize(slsqp_obj, v0_pert, method='SLSQP', bounds=bounds_slsqp,
                           constraints={'type': 'ineq', 'fun': slsqp_cons},
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if np.min(slsqp_cons(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP verification to ensure radii exactly match centers
    final_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_sum:
        best_r = final_r
        best_sum = final_s
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
