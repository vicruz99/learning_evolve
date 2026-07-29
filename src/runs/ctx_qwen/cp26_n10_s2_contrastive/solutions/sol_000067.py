# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state 808bce88) state=51772ca0 sum of radii=1.691450 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
    bounds_r = [(0.0, max(0.0, min(c[0], 1.0 - c[0], c[1], 1.0 - c[1]))) for c in centers]
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method=method)
            if res.success:
                return res.x, -res.fun
        except Exception:
            pass
    return np.zeros(n), 0.0

def obj_joint(x):
    return -np.sum(x[2 * N:])

def constr_joint(x):
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]
    c = np.empty(4 * N + N * (N - 1) // 2)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def gen_inits():
    inits = []
    for s in range(12):
        rng = np.random.RandomState(s)
        c = np.zeros((N, 2))
        idx = 0
        row = 0
        sp = 0.15 + s * 0.006
        while idx < N:
            y = 0.05 + row * sp * math.sqrt(3) / 2
            x0 = 0.05 + (row % 2) * sp / 2
            col = 0
            while x0 + col * sp <= 0.95 and idx < N:
                c[idx] = [x0 + col * sp, y]
                idx += 1
                col += 1
            row += 1
        c += rng.randn(N, 2) * 0.01
        c = np.clip(c, 0.02, 0.98)
        r0 = np.full(N, 0.06)
        inits.append(np.concatenate([c.ravel(), r0]))
        
    for s in range(8):
        rng = np.random.RandomState(s + 100)
        c = rng.uniform(0.1, 0.9, (N, 2))
        r0 = np.full(N, 0.06)
        inits.append(np.concatenate([c.ravel(), r0]))
    return inits

def fix_violations(centers, radii):
    n = centers.shape[0]
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max(0.0, mx - 1e-9))
        
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def run_packing():
    best_sum = 0.0
    best_c = None
    best_r = None
    cons = {'type': 'ineq', 'fun': constr_joint}
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    for x0 in gen_inits():
        try:
            res = minimize(obj_joint, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            c_opt = res.x[:2 * N].reshape(N, 2)
        except Exception:
            continue
            
        r_lp, s_lp = solve_lp_radii(c_opt)
        if s_lp > best_sum:
            best_sum = s_lp
            best_c = c_opt.copy()
            best_r = r_lp.copy()
            
            for _ in range(4):
                rng = np.random.RandomState()
                c_pert = best_c + rng.randn(N, 2) * 0.004
                c_pert = np.clip(c_pert, 0.02, 0.98)
                r_pert = best_r + rng.randn(N) * 0.001
                r_pert = np.clip(r_pert, 0.001, 0.5)
                x0_p = np.concatenate([c_pert.ravel(), r_pert])
                
                try:
                    res2 = minimize(obj_joint, x0_p, method='SLSQP', bounds=bounds, constraints=cons,
                                    options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                    c_opt2 = res2.x[:2 * N].reshape(N, 2)
                    r_lp2, s_lp2 = solve_lp_radii(c_opt2)
                    if s_lp2 > best_sum:
                        best_sum = s_lp2
                        best_c = c_opt2.copy()
                        best_r = r_lp2.copy()
                except Exception:
                    pass
                    
    if best_c is None:
        best_c = np.random.rand(N, 2) * 0.6 + 0.2
        best_r, best_sum = solve_lp_radii(best_c)
        
    best_c, best_r = fix_violations(best_c, best_r)
    return best_c, best_r, float(np.sum(best_r))
