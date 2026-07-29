# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000133 (state 27fd9551) state=2a07c1f5 sum of radii=2.613549 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp_and_gradient(centers):
    """Solves LP for max sum of radii given fixed centers and computes dual gradient."""
    c = np.clip(centers, 1e-9, 1.0 - 1e-9)
    n = centers.shape[0]
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    num_pairs = n * (n - 1) // 2
    num_bnd = 4 * n
    A_ub = np.zeros((num_pairs + num_bnd, n))
    b_ub = np.zeros(num_pairs + num_bnd)
    
    idx = 0
    pair_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            pair_idx[(i, j)] = idx
            idx += 1
            
    for i in range(n):
        x, y = c[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0.0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, None, None
        
    radii = res.x
    duals = np.maximum(res.ineqlin.marginals, 0.0)
    
    grad = np.zeros_like(c)
    
    for i in range(n):
        for j in range(i + 1, n):
            lam = duals[pair_idx[(i, j)]]
            if lam > 1e-8:
                d = dists[i, j]
                if d > 1e-9:
                    vec = (c[i] - c[j]) / d
                    grad[i] += lam * vec
                    grad[j] -= lam * vec
                    
    start_bnd = num_pairs
    for i in range(n):
        mu_L = duals[start_bnd + 4*i]
        mu_R = duals[start_bnd + 4*i + 1]
        mu_B = duals[start_bnd + 4*i + 2]
        mu_T = duals[start_bnd + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return radii, -res.fun, grad

def gradient_ascent(centers0, steps=600, init_step=0.005, rng=None):
    """Moves centers along LP dual gradients to maximize sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    
    step = init_step
    no_improve = 0
    
    for k in range(steps):
        radii, curr_sum, grad = solve_lp_and_gradient(centers)
        if radii is None:
            break
            
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 40:
            step *= 0.6
        elif no_improve > 15:
            step *= 0.8
            
        if step < 1e-10:
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            centers += step * grad / grad_norm
            centers = np.clip(centers, 0.01, 0.99)
            
        if k % 200 == 0 and k > 0:
            centers += rng.normal(0, 0.001, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            
    return best_centers, best_sum

def joint_slsqp(centers0, radii0):
    """Polishes configuration using constrained joint optimization."""
    def obj(v):
        return -np.sum(v[2*N:])
        
    def cons(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        out = []
        out.append(c[:, 0] - r)
        out.append(1.0 - c[:, 0] - r)
        out.append(c[:, 1] - r)
        out.append(1.0 - c[:, 1] - r)
        
        idx_i, idx_j = np.triu_indices(N, 1)
        dx = c[idx_i, 0] - c[idx_j, 0]
        dy = c[idx_i, 1] - c[idx_j, 1]
        dr = r[idx_i] + r[idx_j]
        out.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(out)
        
    v0 = np.concatenate([centers0.flatten(), radii0])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 4000, 'ftol': 1e-12})
        if np.min(cons(res.x)) >= -1e-8:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return centers0, radii0, np.sum(radii0)

def hex_pattern(pattern, r_est=0.095, rng=None):
    """Generates a hexagonal lattice configuration based on row counts."""
    centers = []
    y = r_est
    for row_idx, cnt in enumerate(pattern):
        shift = r_est if row_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            centers.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
    return np.clip(np.array(centers[:N]), 0.05, 0.95)

def generate_starts(rng):
    """Creates diverse initial configurations for multi-start optimization."""
    starts = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], 
        [6,6,5,5,4], [4,6,6,6,4], [5,5,6,5,5],
        [6,5,5,5,5], [5,6,4,6,5], [4,5,6,5,6]
    ]
    for pat in patterns:
        for r in [0.092, 0.096, 0.100, 0.104]:
            starts.append(hex_pattern(pat, r, rng))
            
    for _ in range(8):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(400):
            f = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist = np.maximum(dist, 1e-4)
            f += np.sum(diff / (dist**2)[:, :, None], axis=1)
            c += 0.002 * f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    starts = generate_starts(rng)
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    # Phase 1: Gradient ascent on centers from multiple starts
    for c0 in starts:
        c_opt, s_opt = gradient_ascent(c0, steps=500, init_step=0.004, rng=rng)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            r_opt, _, _ = solve_lp_and_gradient(best_c)
            best_r = r_opt.copy()
            
    # Phase 2: SLSQP joint polish
    if best_c is not None:
        c_polish, r_polish, s_polish = joint_slsqp(best_c, best_r)
        if s_polish > best_sum:
            best_sum = s_polish
            best_c = c_polish
            best_r = r_polish
            
        # Perturb and polish to escape shallow local minima
        for _ in range(4):
            c_pert = best_c + rng.normal(0, 0.0015, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, _, _ = solve_lp_and_gradient(c_pert)
            c2, r2, s2 = joint_slsqp(c_pert, r_pert)
            if s2 > best_sum:
                best_sum = s2
                best_c = c2
                best_r = r2

    # Phase 3: Strict Numerical Repair for validator compliance
    centers = best_c.copy()
    radii = best_r.copy()
    
    for _ in range(60):
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
