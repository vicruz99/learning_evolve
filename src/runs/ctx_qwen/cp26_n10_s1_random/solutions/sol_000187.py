# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=7cba18f0 sum of radii=2.627679 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_hex_config(n, rows, r_init):
    """Generates a hexagonal lattice initialization with specified row counts."""
    pts = []
    y = r_init
    for i, cnt in enumerate(rows):
        shift = r_init if i % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(cnt):
            if len(pts) < n:
                pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
    return np.array(pts[:n])

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    lim = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                     np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lim = np.maximum(lim, 1e-9)
    
    c_obj = -np.ones(n)
    bounds = [(0.0, lim[i]) for i in range(n)]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def joint_objective(vars_array, n):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(vars_array[2*n:])

def joint_constraints(vars_array, n):
    """Returns inequality constraints >= 0 for valid packing."""
    c = vars_array[:2*n].reshape(n, 2)
    r = vars_array[2*n:]
    
    bc = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.concatenate([bc, d2[mask] - r_sum_sq[mask]])

def hill_climb_lp(centers, n, steps=500, rng=None):
    """Derivative-free hill climbing optimizing centers by LP-evaluated radii sum."""
    if rng is None:
        rng = np.random.default_rng()
    best_c = centers.copy()
    r_best, val_best = solve_radii_lp(best_c)
    
    for _ in range(steps):
        i = rng.integers(n)
        step = rng.normal(0, 0.008, 2)
        trial_c = best_c.copy()
        trial_c[i] = np.clip(trial_c[i] + step, 0.001, 0.999)
        
        r_trial, val_trial = solve_radii_lp(trial_c)
        if val_trial > val_best + 1e-7:
            best_c = trial_c
            r_best = r_trial
            val_best = val_trial
            
    return best_c, r_best, val_best

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 5, 5, 6], [5, 4, 6, 5, 6], [6, 4, 5, 6, 5]
    ]
    
    for pat in patterns:
        if sum(pat) < n: continue
        for r0 in [0.095, 0.10, 0.105]:
            cfg = get_hex_config(n, pat, r0)
            configs.append(np.clip(cfg, 0.05, 0.95))
            
    rng = np.random.default_rng(42)
    for _ in range(15):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.2)] * n
    cons_dict = {'type': 'ineq', 'fun': joint_constraints, 'args': (n,)}
    
    # 2. Joint SLSQP Optimization Phase
    for cfg in configs:
        r0 = np.full(n, 0.08)
        x0 = np.concatenate([cfg.flatten(), r0])
        
        try:
            res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                          bounds=bounds_vars, constraints=cons_dict,
                          options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_lp, s_lp = solve_radii_lp(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass
            
    # 3. LP-Driven Hill Climbing Phase (Highly effective for packing polishing)
    if best_centers is not None:
        c_hc, r_hc, s_hc = hill_climb_lp(best_centers, n, steps=600, rng=rng)
        if s_hc > best_sum:
            best_sum = s_hc
            best_centers = c_hc.copy()
            best_radii = r_hc.copy()
            
        # 4. Perturbation + Joint Refinement to escape any remaining local minima
        for _ in range(12):
            pert = best_centers + rng.uniform(-0.004, 0.004, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r0 = np.full(n, 0.09)
            x0 = np.concatenate([pert.flatten(), r0])
            try:
                res = minimize(joint_objective, x0, method='SLSQP', args=(n,),
                              bounds=bounds_vars, constraints=cons_dict,
                              options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun):
                    c_opt = res.x[:2*n].reshape(n, 2)
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass
                
    # Fallback safety net
    if best_centers is None:
        best_centers = np.clip(get_hex_config(n, [5, 6, 5, 6, 4], 0.09), 0.1, 0.9)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # 5. Final safety scaling to strictly satisfy 1e-12 validator tolerance
    scale = 1.0
    c = best_centers
    r = best_radii
    for i in range(n):
        if r[i] > 1e-12:
            scale = min(scale, c[i,0]/r[i], (1.0-c[i,0])/r[i], c[i,1]/r[i], (1.0-c[i,1])/r[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i]-c[j])
            r_sum = r[i] + r[j]
            if r_sum > 1e-12:
                scale = min(scale, d / r_sum)
                
    r *= scale * 0.9999995
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum
