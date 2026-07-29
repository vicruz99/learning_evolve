# sol_000210 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000169 (state 623e904f) state=dd9832a7 sum of radii=2.619856 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

# Precompute LP constraint matrix structure
A_ub_global = np.zeros((N*(N-1)//2 + 4*N, N))
pair_idx_global = []
k = 0
for i in range(N):
    for j in range(i+1, N):
        A_ub_global[k, i] = 1.0
        A_ub_global[k, j] = 1.0
        pair_idx_global.append((i, j))
        k += 1
for i in range(N):
    for _ in range(4):
        A_ub_global[k, i] = 1.0
        k += 1

def solve_lp_and_grad(centers):
    """Solves LP for optimal radii given fixed centers and computes exact gradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b = np.zeros(A_ub_global.shape[0])
    k = 0
    for i, j in pair_idx_global:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_ub_global, b_ub=b, 
                  bounds=[(0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    try:
        duals = np.asarray(res.ineqlin.marginals)
    except AttributeError:
        duals = np.zeros(A_ub_global.shape[0])
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in pair_idx_global:
        lam = duals[k]
        if lam > 1e-7:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        k += 1
        
    bound_start = len(pair_idx_global)
    for i in range(N):
        grad[i, 0] += duals[bound_start + 4*i] - duals[bound_start + 4*i + 1]
        grad[i, 1] += duals[bound_start + 4*i + 2] - duals[bound_start + 4*i + 3]
        
    return radii, np.sum(radii), grad

def gradient_ascent(c0, max_iter=800, rng=None):
    """Performs gradient ascent on centers using LP duals with adaptive line search."""
    c = c0.copy()
    best_c = c.copy()
    best_s = -1.0
    step = 0.015
    
    radii, s, grad = solve_lp_and_grad(c)
    best_s = s
    
    for it in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
            
        g_dir = grad / g_norm
        alpha = step
        improved = False
        for _ in range(10):
            c_new = np.clip(c + alpha * g_dir, 0.01, 0.99)
            _, s_new, _ = solve_lp_and_grad(c_new)
            if s_new > s + 1e-12:
                c = c_new
                s = s_new
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            step *= 0.8
            if step < 1e-9:
                break
        else:
            step = min(step * 1.15, 0.05)
            
        radii, s, grad = solve_lp_and_grad(c)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            
        if rng is not None and it % 150 == 149:
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.02, 0.98)
            radii, s, grad = solve_lp_and_grad(c)
            
    return best_c, best_s

def generate_starts(rng):
    """Generates diverse initial configurations (hexagonal & force-directed)."""
    starts = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [6,4,6,5,5],
        [4,6,6,6,4], [5,4,6,6,5], [6,6,5,5,4], [5,5,6,5,5],
        [4,5,6,5,6], [5,6,4,5,6], [5,5,4,6,6], [6,5,5,5,5],
        [5,5,5,6,5], [4,5,5,6,6], [6,6,4,5,5], [5,7,5,5,4]
    ]
    for pat in patterns:
        for r0 in [0.088, 0.093, 0.098, 0.103]:
            c = []
            y = r0
            for r_idx, cnt in enumerate(pat):
                shift = r0 if r_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            c = np.array(c[:N])
            c += rng.normal(0, 0.003, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for seed in range(15):
        rng_fd = np.random.default_rng(seed)
        c = rng_fd.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i]-c[j])
                    if d < 0.22 and d > 1e-4:
                        f = (0.22 - d) / d
                        diff = c[i] - c[j]
                        forces[i] += diff * f
                        forces[j] -= diff * f
            c += forces * 0.012
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def slsqp_obj(v):
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    dx = c[i, 0] - c[j, 0]
    dy = c[i, 1] - c[j, 1]
    dr = r[i] + r[j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def polish(centers):
    """Refines centers and radii jointly using SLSQP."""
    radii, s, _ = solve_lp_and_grad(centers)
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 8000, 'ftol': 1e-14})
        if np.min(slsqp_cons(res.x)) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], np.sum(res.x[2*N:])
    except Exception:
        pass
    return centers, radii, s

def repair(centers, radii):
    """Deterministic repair to ensure strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
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
    best_c, best_r, best_s = None, None, -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Gradient Ascent from diverse starts
    for c0 in starts:
        c_opt, s_opt = gradient_ascent(c0, rng=rng)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Iterative perturbation to escape local minima
    if best_c is not None:
        for _ in range(25):
            c_pert = best_c + rng.normal(0, 0.005, best_c.shape)
            c_pert = np.clip(c_pert, 0.05, 0.95)
            c_opt, s_opt = gradient_ascent(c_pert, max_iter=600, rng=rng)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                
        # Phase 3: SLSQP Joint Polish
        c_pol, r_pol, s_pol = polish(best_c)
        if s_pol > best_s:
            best_s = s_pol
            best_c = c_pol
            best_r = r_pol
        else:
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Final strict repair
    radii = repair(best_c, best_r.copy())
    return best_c, radii, float(np.sum(radii))
