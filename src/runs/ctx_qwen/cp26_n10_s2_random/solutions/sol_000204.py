# sol_000204 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=3d654943 sum of radii=2.289569 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

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
    ub = np.maximum(ub, 1e-9)
    
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
            return np.zeros(N), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    # Handle different scipy versions for marginals
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except AttributeError:
            duals = np.zeros(A_LP.shape[0])
            
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        lam = duals[k]
        if lam > 1e-7:
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

def optimize_centers(c0, max_iter=1500, rng=None):
    """Performs gradient ascent on centers using LP duals."""
    c = c0.copy()
    best_c = c.copy()
    best_sum = -1.0
    
    radii, s, grad = solve_lp_and_grad(c)
    best_sum = s
    
    step = 0.006
    fail_count = 0
    for k in range(max_iter):
        radii, s, grad = solve_lp_and_grad(c)
        if s > best_sum:
            best_sum = s
            best_c = c.copy()
            fail_count = 0
            
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
            
        g_dir = grad / g_norm
        c_new = c + step * g_dir
        c_new = np.clip(c_new, 0.002, 0.998)
        
        _, s_new, _ = solve_lp_and_grad(c_new)
        
        if s_new > s + 1e-12:
            c = c_new
            step *= 1.07
            step = min(step, 0.028)
            fail_count = 0
        else:
            step *= 0.75
            fail_count += 1
            
        if step < 1e-10 or fail_count > 25:
            break
            
        # Periodic jitter to escape local minima
        if rng is not None and k % 350 == 349:
            c += rng.normal(0, 0.0025, c.shape)
            c = np.clip(c, 0.02, 0.98)
            
    return best_c, best_sum

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [5, 6, 4, 5, 6],
        [4, 5, 6, 5, 6], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [7, 5, 6, 5, 3], [5, 7, 5, 6, 3], [4, 7, 6, 6, 3],
        [6, 6, 6, 4, 4], [5, 5, 7, 5, 4], [6, 5, 7, 5, 3]
    ]
    for pat in patterns:
        for r_est in [0.088, 0.095, 0.102, 0.110]:
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
            
    # Random starts
    for _ in range(15):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Force-relaxed random starts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i]-c[j])
                    if d < 0.22 and d > 1e-4:
                        f = (0.22 - d) / d * 0.012
                        diff = c[i] - c[j]
                        forces[i] += diff * f
                        forces[j] -= diff * f
            c += forces
            c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    return inits

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
                       options={'maxiter': 5000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-9:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], np.sum(res.x[2 * N:])
    except Exception:
        pass
    return centers, radii, s

def repair(centers, radii):
    """Deterministic repair to ensure strict validity."""
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start Gradient Ascent
    starts = generate_inits(rng)
    for c_init in starts:
        c_opt, s_opt = optimize_centers(c_init, max_iter=1800, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Basin-Hopping Perturbations
    if best_c is not None:
        for _ in range(25):
            c_pert = best_c.copy()
            # Randomly displace a subset of circles
            n_pert = rng.integers(2, 6)
            idx = rng.choice(N, n_pert, replace=False)
            c_pert[idx] += rng.normal(0, 0.015, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.04, 0.96)
            
            c_opt2, s_opt2 = optimize_centers(c_pert, max_iter=800, rng=rng)
            if s_opt2 > best_sum:
                best_sum = s_opt2
                best_c = c_opt2.copy()
                
    # Phase 3: SLSQP Joint Polish
    if best_c is not None:
        c_pol, r_pol, s_pol = polish_slsqp(best_c)
        if s_pol > best_sum:
            best_sum = s_pol
            best_c = c_pol
            best_r = r_pol
            
        # Additional perturbation around SLSQP result
        for _ in range(10):
            c_pert3 = best_c + rng.normal(0, 0.008, best_c.shape)
            c_pert3 = np.clip(c_pert3, 0.04, 0.96)
            c_opt3, s_opt3 = optimize_centers(c_pert3, max_iter=600, rng=rng)
            if s_opt3 > best_sum:
                best_sum = s_opt3
                best_c = c_opt3.copy()
                
        c_pol2, r_pol2, s_pol2 = polish_slsqp(best_c)
        if s_pol2 > best_sum:
            best_sum = s_pol2
            best_c = c_pol2
            best_r = r_pol2

    # Final LP solve for exact radii matching best centers
    if best_c is not None:
        best_r, final_s, _ = solve_lp_and_grad(best_c)
        if final_s > best_sum:
            best_sum = final_s
            
    centers = best_c.copy()
    radii = repair(centers, best_r.copy())
    
    return centers, radii, float(np.sum(radii))
