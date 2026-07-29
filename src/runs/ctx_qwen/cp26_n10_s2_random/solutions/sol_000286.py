# sol_000286 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000227 (state 324f8d76) state=d00da21c sum of radii=2.630700 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii, duals, and sum."""
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
        # r_i <= x_i
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 0]; idx += 1
        # r_i <= 1 - x_i
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        # r_i <= y_i
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 1]; idx += 1
        # r_i <= 1 - y_i
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
        if not res.success:
            return np.full(n, 1e-4), np.zeros_like(b_ub), 0.0
            
        radii = res.x
        sum_r = np.sum(radii)
        
        # Extract duals safely
        duals = np.zeros_like(b_ub)
        try:
            duals = res.marginals.ineqlin
        except AttributeError:
            try:
                duals = res.ineqlin.marginals
            except AttributeError:
                pass
        return radii, duals, sum_r
    except Exception:
        return np.full(n, 1e-4), np.zeros(num_pairs + num_bound), 0.0

def lp_obj_and_grad(centers_flat):
    """Objective and exact gradient for L-BFGS-B optimization of centers."""
    centers = centers_flat.reshape(N, 2)
    radii, duals, sum_r = solve_lp(centers)
    
    obj = -sum_r
    grad = np.zeros_like(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    
    num_pairs = N * (N - 1) // 2
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            lam = duals[idx]
            if lam > 1e-8:
                d = dists[i, j]
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
            idx += 1
            
    for i in range(N):
        mu_L = duals[num_pairs + 4 * i]
        mu_R = duals[num_pairs + 4 * i + 1]
        mu_B = duals[num_pairs + 4 * i + 2]
        mu_T = duals[num_pairs + 4 * i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    # We minimize -sum(radii), so gradient sign is flipped
    return obj, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], 
        [4, 5, 6, 5, 6], [5, 4, 6, 6, 5], [6, 5, 5, 6, 4],
        [5, 5, 4, 6, 6], [4, 6, 5, 5, 6], [6, 4, 5, 6, 5]
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
                        c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            while len(c) < N:
                c.append(rng.uniform(0.2, 0.8, 2))
            starts.append(np.array(c[:N]))
            
    # Force-directed spreads
    for _ in range(15):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(800):
            forces = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-6:
                        push = (0.22 - d) * 0.04
                        forces[i] += d_vec / d * push
                        forces[j] -= d_vec / d * push
            c += forces
            c = np.clip(c, 0.1, 0.9)
        starts.append(c)
        
    return starts

def obj_joint(v):
    return -np.sum(v[2 * N:])

def cons_joint_sq(v):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.02, 0.98)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c0 in starts:
        try:
            res = minimize(lp_obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 3000, 'ftol': 1e-13})
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
    best_r, _, best_sum = solve_lp(best_c)
    
    # Phase 2: Simulated Annealing Perturbation
    T = 0.008
    for step in range(60):
        noise_scale = 0.007 * (1.0 + 0.5 * np.exp(-step / 10.0))
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, 0.03, 0.97)
        
        try:
            res = minimize(lp_obj_and_grad, c_pert.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 1500, 'ftol': 1e-13})
            c_pert = res.x.reshape(N, 2)
            r_pert, _, s_pert = solve_lp(c_pert)
            
            delta = s_pert - best_sum
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
                best_sum = s_pert
                best_c = c_pert.copy()
                best_r = r_pert.copy()
                if delta > 0:
                    T = min(T * 1.05, 0.02)
                else:
                    T *= 0.96
        except Exception:
            continue
            
    # Phase 3: Joint SLSQP Polish
    v0 = np.concatenate([best_c.flatten(), best_r])
    for _ in range(3):
        try:
            res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slsqp,
                          constraints={'type': 'ineq', 'fun': cons_joint_sq},
                          options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(cons_joint_sq(res.x)) >= -1e-8:
                s = np.sum(res.x[2 * N:])
                if s > best_sum:
                    best_sum = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
                    v0 = res.x.copy()
        except Exception:
            pass
            
    # Final LP verification
    lp_r, _, lp_sum = solve_lp(best_c)
    if lp_sum > best_sum:
        best_r = lp_r
        best_sum = lp_sum
        
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
