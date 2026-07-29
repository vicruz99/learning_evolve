# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000124 (state e4120b9c) state=23ddeaa2 sum of radii=2.074127 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(PAIRS)

def solve_lp_and_grad(centers):
    """Solves LP for radii and computes gradient of sum of radii w.r.t centers."""
    c = np.clip(centers, 1e-7, 1.0 - 1e-7)
    n = c.shape[0]
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    num_con = NUM_PAIRS + 4 * n
    A_ub = np.zeros((num_con, n))
    b_ub = np.zeros(num_con)
    
    idx = 0
    for i, j in PAIRS:
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dists[i, j]
        idx += 1
        
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = c[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - c[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = c[i, 1]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - c[i, 1]; idx += 1
        
    c_obj = -np.ones(n)
    bounds_r = [(0.0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return -10.0, np.zeros(n), np.zeros((n, 2))
        
    radii = res.x
    
    # Extract duals safely across scipy versions
    try:
        duals = np.asarray(res.ineqlin.marginals)
    except AttributeError:
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            duals = np.zeros(num_con)
            
    grad = np.zeros((n, 2))
    idx = 0
    for i, j in PAIRS:
        lam = duals[idx]
        if lam > 1e-9:
            d = dists[i, j]
            if d < 1e-10: d = 1e-10
            vec = c[i] - c[j]
            f = lam / d
            grad[i] += f * vec
            grad[j] -= f * vec
        idx += 1
        
    idx = NUM_PAIRS
    for i in range(n):
        mu_L = duals[idx]; idx += 1
        mu_R = duals[idx]; idx += 1
        mu_B = duals[idx]; idx += 1
        mu_T = duals[idx]; idx += 1
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T
        
    return -res.fun, radii, grad

def obj_func(x_flat):
    c = x_flat.reshape(N, 2)
    val, _, _ = solve_lp_and_grad(c)
    return -val

def jac_func(x_flat):
    c = x_flat.reshape(N, 2)
    _, _, grad = solve_lp_and_grad(c)
    return -grad.flatten()

def get_initial_configs(rng):
    configs = []
    patterns = [[5,5,5,5,6], [6,5,6,5,4], [5,6,5,6,4], [7,6,6,7], [6,6,5,5,4], [5,5,5,5,5,1]]
    for pat in patterns:
        pts = []
        r_est = 0.095
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        c = np.array(pts[:N])
        c = (c - c.min(axis=0)) / (c.max(axis=0) - c.min(axis=0)) * 0.9 + 0.05
        configs.append(c)
        
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        configs.append(c)
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0)] * (2 * N)
    
    best_val = -10.0
    best_centers = None
    best_radii = None
    
    starts = get_initial_configs(rng)
    
    for c_init in starts:
        c_pert = c_init + rng.normal(0, 0.004, c_init.shape)
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        x0 = c_pert.flatten()
        
        try:
            res = minimize(obj_func, x0, jac=jac_func, method='L-BFGS-B', 
                          bounds=bounds_opt, options={'maxiter': 2500, 'ftol': 1e-13})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_centers = res.x.reshape(N, 2)
                _, best_radii, _ = solve_lp_and_grad(best_centers)
        except Exception:
            pass
            
    if best_centers is not None:
        for step in range(40):
            noise = 0.006 * (0.82 ** step)
            c_trial = best_centers + rng.normal(0, noise, best_centers.shape)
            c_trial = np.clip(c_trial, 0.01, 0.99)
            
            try:
                res = minimize(obj_func, c_trial.flatten(), jac=jac_func, method='L-BFGS-B',
                              bounds=bounds_opt, options={'maxiter': 2000, 'ftol': 1e-13})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_centers = res.x.reshape(N, 2)
                    _, best_radii, _ = solve_lp_and_grad(best_centers)
            except Exception:
                pass
                
    centers = best_centers
    radii = best_radii
    
    # Strict numerical repair
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
