# sol_000287 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000227 (state 324f8d76) state=4c08251c sum of radii=2.634292 correctness=1.0
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
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
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
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
        if res.success:
            try:
                duals = res.marginals.ineqlin
            except AttributeError:
                try:
                    duals = res.ineqlin.marginals
                except AttributeError:
                    duals = np.zeros_like(b_ub)
            return res.x, duals
    except Exception:
        pass
    # Fallback feasible radii
    r_fallback = np.ones(n) * 0.04
    return r_fallback, np.zeros(num_pairs + num_bound)

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

def lp_obj_and_grad(centers_flat):
    """Objective and exact gradient for L-BFGS-B optimization of centers."""
    centers = centers_flat.reshape(N, 2)
    radii, duals = solve_lp(centers)
    obj = -np.sum(radii)
    grad = compute_grad(centers, duals)
    # We minimize -sum(radii), so gradient is -grad
    return obj, -grad.flatten()

def generate_inits(rng):
    """Generates diverse initial configurations."""
    inits = []
    
    # Hexagonal patterns with varying row counts
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
        [5, 4, 6, 6, 5], [4, 5, 6, 5, 6], [6, 5, 5, 6, 4]
    ]
    for pat in patterns:
        for r_est in [0.098, 0.102, 0.105, 0.108]:
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
            inits.append(c)
            
    # Corner/Edge biased starts
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        c[:4] = corners
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
        inits.append(c)
        
    # Force-directed random spreads
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.25 and d > 1e-6:
                        push = (0.25 - d) * 0.05
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        inits.append(c)
        
    return inits

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
    bounds_lbfgs = [(0.01, 0.99)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Multi-start L-BFGS-B Center Optimization
    inits = generate_inits(rng)
    for c0 in inits:
        try:
            res = minimize(lp_obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 3000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, _ = solve_lp(c_opt)
            s_opt = np.sum(r_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
        best_r, _ = solve_lp(best_c)
        best_sum = np.sum(best_r)
        
    # Phase 2: Basin-Hopping Shake to Escape Local Minima
    for step in range(40):
        scale = 0.009 * (0.88 ** (step // 4))
        c_trial = best_c + rng.normal(0, scale, best_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        try:
            res = minimize(lp_obj_and_grad, c_trial.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, _ = solve_lp(c_opt)
            s_opt = np.sum(r_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP Joint Polish for Precision
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(4):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': cons_joint_sq},
                          options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if np.min(cons_joint_sq(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Final LP Verification & Strict Repair
    lp_r, _ = solve_lp(best_c)
    if np.sum(lp_r) > best_sum:
        best_r = lp_r
        best_sum = np.sum(lp_r)
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
