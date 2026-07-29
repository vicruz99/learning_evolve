# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state 808bce88) state=1d8b90a2 sum of radii=1.678570 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
N_PAIRS = len(PAIR_I)

def get_optimal_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((n * (n - 1) // 2, n))
    b_ub = np.zeros(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.hypot(dx, dy)
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dist
            k += 1
            
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        if ub < 0.0:
            ub = 0.0
        bounds.append((0.0, ub))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-8):
                return res.x, -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0

def obj_func(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * N:])

def constr_func(x):
    """Inequality constraints: boundary and pairwise non-overlap."""
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]
    
    c = np.empty(4 * N + N_PAIRS)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    
    dx = cx[PAIR_I] - cx[PAIR_J]
    dy = cy[PAIR_I] - cy[PAIR_J]
    dists = np.sqrt(dx * dx + dy * dy)
    c[4 * N:] = dists - (r[PAIR_I] + r[PAIR_J])
    
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constr_func}
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with varying spacing and margins
    for seed in range(20):
        rng = np.random.RandomState(seed)
        centers = np.zeros((N, 2))
        idx = 0
        row = 0
        spacing = 0.155 + (seed % 10) * 0.004
        margin = 0.04 + (seed // 10) * 0.02
        while idx < N:
            y = margin + row * spacing * math.sqrt(3) / 2
            if y > 1.0 - margin:
                break
            x_start = margin + (row % 2) * spacing / 2
            col = 0
            while x_start + col * spacing <= 1.0 - margin and idx < N:
                centers[idx] = [x_start + col * spacing, y]
                idx += 1
                col += 1
            row += 1
        while idx < N:
            centers[idx] = rng.uniform(0.15, 0.85, 2)
            idx += 1
        centers += rng.randn(N, 2) * 0.012
        centers = np.clip(centers, 0.02, 0.98)
        r_init = np.full(N, 0.05)
        inits.append(np.concatenate([centers.ravel(), r_init]))
        
    # 2. Grid layouts with perturbation
    for seed in range(10):
        rng = np.random.RandomState(1000 + seed)
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < N:
                    centers[idx] = [0.08 + j * 0.18, 0.08 + i * 0.15]
                    idx += 1
        while idx < N:
            centers[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
        centers += rng.randn(N, 2) * 0.015
        centers = np.clip(centers, 0.02, 0.98)
        r_init = np.full(N, 0.05)
        inits.append(np.concatenate([centers.ravel(), r_init]))

    # Main optimization loop
    for x0 in inits:
        try:
            res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            
            c_opt = res.x[:2 * N].reshape(N, 2)
            # Clamp strictly inside for LP stability
            c_polish = np.clip(c_opt, 1e-5, 1.0 - 1e-5)
            r_lp, sum_lp = get_optimal_radii(c_polish)
            
            if sum_lp > best_sum:
                best_sum = sum_lp
                best_centers = c_polish.copy()
                best_radii = r_lp.copy()
                
                # Refine centers from LP result
                x0_ref = np.concatenate([best_centers.ravel(), best_radii * 0.995])
                res2 = minimize(obj_func, x0_ref, method='SLSQP', bounds=bounds, constraints=cons,
                                options={'maxiter': 4000, 'ftol': 1e-13})
                c_opt2 = res2.x[:2 * N].reshape(N, 2)
                c_opt2 = np.clip(c_opt2, 1e-5, 1.0 - 1e-5)
                r_lp2, sum_lp2 = get_optimal_radii(c_opt2)
                if sum_lp2 > best_sum:
                    best_sum = sum_lp2
                    best_centers = c_opt2.copy()
                    best_radii = r_lp2.copy()
        except Exception:
            continue
            
    # Local perturbation search on best found configuration
    if best_centers is not None:
        for trial in range(40):
            rng = np.random.RandomState(trial * 7 + 13)
            c_pert = best_centers + rng.randn(N, 2) * 0.004
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_p, s_p = get_optimal_radii(c_pert)
            
            if s_p > best_sum:
                best_sum = s_p
                best_centers = c_pert.copy()
                best_radii = r_p.copy()
                
                x0_p = np.concatenate([best_centers.ravel(), best_radii])
                res_p = minimize(obj_func, x0_p, method='SLSQP', bounds=bounds, constraints=cons,
                                 options={'maxiter': 2500, 'ftol': 1e-13})
                c_opt_p = res_p.x[:2 * N].reshape(N, 2)
                c_opt_p = np.clip(c_opt_p, 1e-5, 1.0 - 1e-5)
                r_lp_p, sum_lp_p = get_optimal_radii(c_opt_p)
                if sum_lp_p > best_sum:
                    best_sum = sum_lp_p
                    best_centers = c_opt_p.copy()
                    best_radii = r_lp_p.copy()

    # Fallback (should not be reached)
    if best_centers is None:
        best_centers = np.tile([0.5, 0.5], (N, 1))
        best_radii = np.zeros(N)
        best_sum = 0.0
        
    # Final strict validation and numerical safety cleanup
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve microscopic overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
                s = r_final[i] + r_final[j]
                if d < s - 1e-11:
                    excess = s - d + 1e-11
                    r_final[i] -= excess / 2.0
                    r_final[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))
