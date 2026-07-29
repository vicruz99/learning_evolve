# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=fbefd06d sum of radii=2.612467 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def setup_lp():
    """Precompute the sparse structure of the LP constraint matrix."""
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A = np.zeros((num_pairs + num_bound, N))
    p_idx = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[k, i] = 1.0
            A[k, j] = 1.0
            p_idx.append((i, j))
            k += 1
    for i in range(N):
        for _ in range(4):
            A[k, i] = 1.0
            k += 1
    return A, p_idx

A_LP, PAIR_IDX = setup_lp()

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.clip(ub, 1e-6, None)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
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
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                      bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.full(N, 0.01), 0.0, np.zeros_like(centers)
    except Exception:
        return np.full(N, 0.01), 0.0, np.zeros_like(centers)
        
    radii = res.x
    try:
        duals = np.asarray(res.ineqlin.marginals)
    except (AttributeError, IndexError):
        duals = np.zeros(A_LP.shape[0])
        
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
        mu_x0 = duals[bound_start + 4 * i]
        mu_x1 = duals[bound_start + 4 * i + 1]
        mu_y0 = duals[bound_start + 4 * i + 2]
        mu_y1 = duals[bound_start + 4 * i + 3]
        grad[i, 0] += mu_x0 - mu_x1
        grad[i, 1] += mu_y0 - mu_y1
        
    return radii, np.sum(radii), grad

def obj_and_grad(x):
    """Objective and gradient for scipy optimizer."""
    c = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def force_directed_init(rng, seed_shift=0):
    """Generates a well-spaced initial configuration using repulsive forces."""
    rng_fd = np.random.default_rng(seed_shift + 42)
    c = rng_fd.uniform(0.1, 0.9, (N, 2))
    for _ in range(1500):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 0.22 and dist > 1e-6:
                    f = (0.22 - dist) / dist
                    forces[i] += diff * f
                    forces[j] -= diff * f
        c += forces * 0.008
        c = np.clip(c, 0.02, 0.98)
    return c

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [5, 6, 4, 5, 6],
        [4, 5, 5, 6, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.088, 0.094, 0.099, 0.104]:
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
            c = np.clip(c, 0.04, 0.96)
            starts.append(c)
            
    for _ in range(10):
        starts.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    for s in range(15):
        starts.append(force_directed_init(rng, seed_shift=s))
        
    return starts

def slsqp_obj(v):
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
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

def polish_slsqp(centers):
    """Refines centers and radii jointly using SLSQP."""
    radii, s, _ = solve_lp_and_grad(centers)
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-8:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], np.sum(res.x[2 * N:])
    except Exception:
        pass
    return centers, radii, s

def repair(centers, radii):
    """Deterministic repair to ensure strict validation compliance."""
    radii = radii.copy()
    for _ in range(80):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-11:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    bounds_c = [(0.005, 0.995)] * (2 * N)
    
    # Phase 1: L-BFGS-B Gradient Ascent from diverse starts
    for c_init in starts:
        try:
            res = minimize(obj_and_grad, c_init.flatten(), method='L-BFGS-B',
                           bounds=bounds_c, jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    # Phase 2: Simulated Annealing Perturbations to escape local minima
    if best_c is not None:
        temp = 0.012
        for step in range(60):
            noise = temp * (0.92 ** step)
            c_pert = best_c + rng.normal(0, noise, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            try:
                res = minimize(obj_and_grad, c_pert.flatten(), method='L-BFGS-B',
                               bounds=bounds_c, jac=True,
                               options={'maxiter': 2000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_sum + 1e-9:
                    best_sum = -res.fun
                    best_c = res.x.reshape(N, 2).copy()
            except Exception:
                pass
                
            # Homotopy growth: slightly inflate radii target and re-optimize
            if step % 10 == 9 and step > 0:
                _, _, g = solve_lp_and_grad(best_c)
                g_norm = np.linalg.norm(g)
                if g_norm > 1e-9:
                    best_c += 0.005 * g / g_norm
                    best_c = np.clip(best_c, 0.02, 0.98)
                    
    # Phase 3: SLSQP Joint Polish
    if best_c is not None:
        c_pol, r_pol, s_pol = polish_slsqp(best_c)
        if s_pol > best_sum:
            best_sum = s_pol
            best_c = c_pol
            best_r = r_pol
            
        # One final aggressive perturbation search on the polished config
        for _ in range(5):
            c_try = best_c + rng.normal(0, 0.004, best_c.shape)
            c_try = np.clip(c_try, 0.05, 0.95)
            c_try, _, _ = polish_slsqp(c_try)
            if np.sum(c_try) > best_sum: # dummy check, rely on polish output
                pass
            _, s_try, _ = solve_lp_and_grad(c_try)
            if s_try > best_sum:
                best_sum = s_try
                best_c = c_try

    # Final LP solve and strict repair
    if best_c is not None:
        best_r, final_s, _ = solve_lp_and_grad(best_c)
        if final_s > best_sum:
            best_sum = final_s
            
    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    
    return centers, radii, float(np.sum(radii))
