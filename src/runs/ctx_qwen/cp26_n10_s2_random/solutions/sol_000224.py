# sol_000224 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000123 (state 101aee21) state=70cd4a3b sum of radii=2.628631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii and dual variables."""
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
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
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
    if res.success:
        return res.x, res.ineqlin.marginals
    return None, None

def compute_grad(centers, duals):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    num_pairs = n * (n - 1) // 2
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
            
    for i in range(n):
        mu_L = duals[num_pairs + 4 * i]
        mu_R = duals[num_pairs + 4 * i + 1]
        mu_B = duals[num_pairs + 4 * i + 2]
        mu_T = duals[num_pairs + 4 * i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return grad

def gradient_ascent(centers0, max_iter=2500):
    """Performs gradient ascent on centers to maximize LP-optimal sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = 0.004
    no_improve = 0
    
    for k in range(max_iter):
        radii, duals = solve_lp(centers)
        if radii is None:
            break
        curr_sum = np.sum(radii)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 70:
            step *= 0.85
        if step < 1e-9:
            break
            
        grad = compute_grad(centers, duals)
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            centers += step * grad / g_norm
            centers = np.clip(centers, 0.015, 0.985)
            
        if k % 150 == 0 and k > 0:
            centers += np.random.normal(0, 0.0015, centers.shape)
            centers = np.clip(centers, 0.015, 0.985)
            
    return best_centers, best_sum

def force_directed_init(seed):
    """Generates an evenly spread initial configuration using repulsive forces."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(1200):
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = c[i] - c[j]
                d = np.linalg.norm(d_vec)
                if d < 0.25 and d > 1e-6:
                    push = (0.25 - d) * 0.06
                    forces[i] += d_vec / d * push
                    forces[j] -= d_vec / d * push
        c += forces
        c = np.clip(c, 0.1, 0.9)
    return c

def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint(v):
    """Constraints for SLSQP: boundaries and non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
    con.append(d - (r[idx[0]] + r[idx[1]]))
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = []
    
    # Hexagonal lattice patterns
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 6, 6, 4],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6], [5, 4, 6, 6, 5]
    ]
    for pat in patterns:
        c_hex = []
        r_est = 0.1
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(c_hex) < N:
                    c_hex.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        while len(c_hex) < N:
            c_hex.append(rng.uniform(0.1, 0.9, 2))
        inits.append(np.array(c_hex[:N]))
        
    # Force-directed spreads
    for s in range(10):
        inits.append(force_directed_init(s))
        
    # Uniform random
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))

    # Phase 1: Gradient Ascent on diverse starts
    for c0 in inits:
        c_ga, s_ga = gradient_ascent(c0)
        if s_ga > best_sum:
            best_sum = s_ga
            best_c = c_ga.copy()
            
    if best_c is None:
        best_c = inits[0]
        best_r, _ = solve_lp(best_c)
        best_sum = np.sum(best_r)
    else:
        best_r, _ = solve_lp(best_c)
        
    # Phase 2: SLSQP Joint Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(3):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_opt,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if np.min(cons_joint(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Perturbation & Refinement Loop
    for step in range(30):
        noise = 0.008 * (0.9 ** (step // 5))
        c_pert = best_c + rng.normal(0, noise, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        c_pert, s_pert = gradient_ascent(c_pert, max_iter=600)
        r_pert, _ = solve_lp(c_pert)
        if r_pert is None:
            continue
            
        v0_pert = np.concatenate([c_pert.flatten(), r_pert])
        try:
            res = minimize(obj_joint, v0_pert, method='SLSQP', bounds=bounds_opt,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 2000, 'ftol': 1e-13, 'disp': False})
            if np.min(cons_joint(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
        except Exception:
            continue
            
    # Final LP verification for optimal radii on best centers
    lp_r, _ = solve_lp(best_c)
    if lp_r is not None and np.sum(lp_r) > best_sum:
        best_r = lp_r
        best_sum = np.sum(lp_r)
        
    # Strict Numerical Repair
    radii = best_r.copy()
    centers = best_c.copy()
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
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
