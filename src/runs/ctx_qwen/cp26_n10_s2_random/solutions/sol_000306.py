# sol_000306 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=e98c8854 sum of radii=2.628190 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N
TOTAL_CONSTRAINTS = NUM_PAIRS + NUM_BOUND

# Precompute constant part of A_ub for LP: r_i + r_j <= dist_ij, r_i <= bounds
A_ub_const = np.zeros((TOTAL_CONSTRAINTS, N))
pair_indices = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_ub_const[idx, i] = 1.0
        A_ub_const[idx, j] = 1.0
        pair_indices.append((i, j))
        idx += 1
for i in range(N):
    A_ub_const[idx + 4*i, i] = 1.0
    A_ub_const[idx + 4*i + 1, i] = 1.0
    A_ub_const[idx + 4*i + 2, i] = 1.0
    A_ub_const[idx + 4*i + 3, i] = 1.0

def solve_lp_and_grad(centers):
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-12)
    
    b_ub = np.zeros(TOTAL_CONSTRAINTS)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    k = 0
    for i, j in pair_indices:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_const, b_ub=b_ub,
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.full(N, 0.01), 0.01 * N, np.zeros((N, 2))
        
        radii = res.x
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except (AttributeError, TypeError):
            duals = np.zeros(TOTAL_CONSTRAINTS)
    except Exception:
        return np.full(N, 0.01), 0.01 * N, np.zeros((N, 2))
        
    grad = np.zeros((N, 2))
    k = 0
    for i, j in pair_indices:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    bound_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return radii, np.sum(radii), grad

def obj_and_grad(x_flat):
    c = np.clip(x_flat.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def generate_inits(rng):
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4],
        [7, 7, 6, 6], [6, 7, 7, 6], [5, 7, 7, 7]
    ]
    for pat in patterns:
        for r_est in [0.095, 0.100, 0.105, 0.110]:
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
            
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            d = c[:, None, :] - c[None, :, :]
            nd = np.linalg.norm(d, axis=2)
            nd = np.maximum(nd, 1e-6)
            w = np.where(nd < 0.2, 1.0 / nd**2, 0.0)
            np.fill_diagonal(w, 0)
            f += np.sum(d * w[:, :, None], axis=1)
            c += 0.002 * f
            c = np.clip(c, 0.1, 0.9)
        inits.append(c)
        
    return inits

def repair(centers, radii):
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d - 1e-12:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-9
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

def obj_joint(v):
    return -np.sum(v[2 * N:])

def cons_joint(v):
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def run_packing():
    rng = np.random.default_rng(42)
    bounds_lb = [(0.001, 0.999)] * (2 * N)
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start L-BFGS-B with exact LP gradient
    for c0 in inits:
        try:
            res = minimize(obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lb, options={'maxiter': 4000, 'ftol': 1e-14})
            c_opt = np.clip(res.x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
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
        
    # Phase 2: Shake & Swap Basin-Hopping to escape local minima
    for step in range(80):
        scale = 0.012 * (0.89 ** (step // 4))
        c_try = best_c.copy()
        
        idx_shuffle = rng.permutation(N)
        c_try[idx_shuffle] += rng.normal(0, scale, c_try.shape)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        swap_idx = rng.choice(N, 2, replace=False)
        c_try[swap_idx] = c_try[swap_idx[::-1]]
        
        try:
            res = minimize(obj_and_grad, c_try.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lb, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = np.clip(res.x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP Joint Polish for precision
    v0 = np.concatenate([best_c.flatten(), best_r * 0.99])
    for _ in range(8):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_sl,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(cons_joint(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP verification
    lp_r, final_s, _ = solve_lp_and_grad(best_c)
    if final_s > best_sum:
        best_r = lp_r
        best_sum = final_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
