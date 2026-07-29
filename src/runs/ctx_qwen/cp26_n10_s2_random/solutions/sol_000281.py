# sol_000281 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000227 (state 324f8d76) state=76dcdda4 sum of radii=2.624544 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii and dual variables."""
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
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
        
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    if not res.success:
        return None, None
        
    try:
        duals = res.marginals.ineqlin
    except AttributeError:
        try:
            duals = res.ineqlin.marginals
        except AttributeError:
            duals = np.zeros(len(b_ub))
            
    return res.x, duals

def compute_grad(centers, duals):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    num_pairs = n * (n - 1) // 2
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            lam = duals[idx]
            if lam > 1e-9:
                d = dists[i, j]
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

def obj_and_grad(v):
    """Objective and exact gradient for L-BFGS-B optimization of centers."""
    c = v.reshape(N, 2)
    radii, duals = solve_lp(c)
    if radii is None:
        return 0.0, np.zeros_like(v)
    val = -np.sum(radii)
    g = compute_grad(c, duals)
    return val, -g.flatten()

def repulsion_init(rng):
    """Generates an evenly spread initial configuration using repulsive forces."""
    c = rng.uniform(0.1, 0.9, (N, 2))
    for _ in range(2000):
        forces = np.zeros_like(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        mask = d < 0.16
        rep = (0.16 - d) / (d**2 + 1e-6)
        rep[~mask] = 0.0
        for i in range(N):
            forces[i] = np.sum(rep[i, :, np.newaxis] * diff[i], axis=0)
        c += forces * 0.006
        c = np.clip(c, 0.05, 0.95)
    return c

def hex_init(rng, pat, r0):
    """Generates a hexagonal lattice initialization with specified row counts."""
    c = []
    y = r0
    for r_idx, cnt in enumerate(pat):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(c) < N:
                c.append([x + rng.normal(0, 0.001), y + rng.normal(0, 0.001)])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
    while len(c) < N:
        c.append(rng.uniform(0.2, 0.8, 2))
    return np.array(c[:N])

def obj_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_joint(v):
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

def repair_packing(centers, radii):
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

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.02, 0.98)] * (2 * N)
    
    best_c = None
    best_sum = -1.0
    
    # Phase 1: Diverse Initialization
    inits = []
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], 
        [4, 5, 6, 5, 6], [5, 4, 6, 6, 5], [6, 5, 5, 6, 4]
    ]
    for pat in patterns:
        for r_est in [0.095, 0.100, 0.105]:
            inits.append(hex_init(rng, pat, r_est))
            
    for _ in range(15):
        inits.append(repulsion_init(rng))
        
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 2: Multi-start L-BFGS-B Optimization
    for c0 in inits:
        try:
            res = minimize(obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 4000, 'ftol': 1e-14})
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        
    # Phase 3: Simulated Annealing Basin Hopping on Centers
    T = 0.006
    for step in range(80):
        noise_scale = 0.009 * (0.94 ** (step // 4))
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        try:
            res = minimize(obj_and_grad, c_pert.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-14})
            s_new = -res.fun
            delta = s_new - best_sum
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-7)):
                best_sum = s_new
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
        T *= 0.975
        
    # Phase 4: Joint SLSQP Polish for Precision
    radii, _ = solve_lp(best_c)
    if radii is not None:
        v0 = np.concatenate([best_c.flatten(), radii])
    else:
        v0 = np.concatenate([best_c.flatten(), np.ones(N) * 0.05])
        
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    for _ in range(6):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': cons_joint},
                          options={'maxiter': 6000, 'ftol': 1e-14})
            if np.min(cons_joint(res.x)) >= -1e-8:
                s_sl = np.sum(res.x[2 * N:])
                if s_sl > best_sum:
                    best_sum = s_sl
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    radii = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 5: Final LP Verification & Strict Repair
    lp_r, _ = solve_lp(best_c)
    if lp_r is not None and np.sum(lp_r) > best_sum:
        radii = lp_r
        best_sum = np.sum(radii)
        
    radii = repair_packing(best_c.copy(), radii.copy())
    return best_c, radii, float(np.sum(radii))
