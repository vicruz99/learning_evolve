# sol_000311 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=c8e1752f sum of radii=2.623499 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact gradient via duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    b_ub = np.zeros(num_pairs + num_bound)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 1]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.zeros(n), 0.0, np.zeros_like(centers)
    except Exception:
        return np.zeros(n), 0.0, np.zeros_like(centers)
        
    radii = res.x
    sum_r = np.sum(radii)
    
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except Exception:
            duals = np.zeros_like(b_ub)
            
    grad = np.zeros_like(centers)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            lam = duals[idx]
            if lam > 1e-8:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (centers[i] - centers[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
            idx += 1
            
    b_start = num_pairs
    for i in range(n):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, sum_r, grad

def lbfgs_obj_grad(x_flat):
    """Objective and exact gradient wrapper for L-BFGS-B."""
    centers = x_flat.reshape(N, 2)
    _, val, grad = solve_lp_and_grad(centers)
    return -val, -grad.flatten()

def optimize_lbfgs(c0, max_iter=5000):
    """Runs L-BFGS-B to optimize centers."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(lbfgs_obj_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [7, 6, 6, 7], [6, 7, 6, 7], [8, 6, 6, 6], [6, 8, 6, 6],
        [7, 7, 6, 6], [6, 6, 7, 7], [5, 7, 7, 7], [7, 7, 7, 5]
    ]
    
    # Hexagonal lattice patterns
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
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    # Force-directed layouts
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(800):
            forces = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            dist = np.maximum(dist, 1e-4)
            mask = dist < 0.22
            rep = np.zeros_like(dist)
            rep[mask] = 0.02 / (dist[mask]**2)
            forces = np.sum(diff / dist[:, :, None] * rep[:, :, None], axis=1)
            c += forces * 0.008
            c = np.clip(c, 0.08, 0.92)
        starts.append(c)
        
    # Corner/Edge biased
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        c += rng.normal(0, 0.015, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Uniform random
    for _ in range(20):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

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

def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint_sq(v):
    """Constraints for SLSQP: boundaries and non-overlap using squared distances."""
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start L-BFGS-B
    inits = generate_starts(rng)
    for c0 in inits:
        c_opt, s_opt = optimize_lbfgs(c0, max_iter=5000)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    if best_c is None:
        best_c = inits[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Decaying perturbation search
    for step in range(60):
        scale = 0.012 * (0.91 ** (step // 3))
        c_trial = best_c + rng.normal(0, scale, best_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        c_opt, s_opt = optimize_lbfgs(c_trial, max_iter=2000)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 3: Focused perturbation (move 1-3 circles)
    for _ in range(40):
        c_trial = best_c.copy()
        n_perturb = rng.integers(1, 4)
        idx_perturb = rng.choice(N, n_perturb, replace=False)
        c_trial[idx_perturb] += rng.normal(0, 0.008, (n_perturb, 2))
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        c_opt, s_opt = optimize_lbfgs(c_trial, max_iter=1500)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 4: SLSQP Joint Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res_sl = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_sl,
                          constraints={'type': 'ineq', 'fun': cons_joint_sq},
                          options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_joint_sq(res_sl.x)) >= -1e-8:
            s_sl = np.sum(res_sl.x[2 * N:])
            if s_sl > best_sum:
                best_sum = s_sl
                best_c = res_sl.x[:2 * N].reshape(N, 2).copy()
                best_r = res_sl.x[2 * N:].copy()
    except Exception:
        pass
        
    # Final LP verification to ensure radii are optimal for final centers
    lp_r, lp_s, _ = solve_lp_and_grad(best_c)
    if lp_s > best_sum:
        best_r = lp_r
        best_sum = lp_s
        
    # Strict numerical repair
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
