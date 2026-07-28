# sol_000209 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000165 (state ab534a56) state=26f2f32e sum of radii=2.104658 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_constraints(vars_arr, n):
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    D = vars_arr[2*n]
    
    con = np.empty(4*n + n*(n-1)//2)
    con[:n] = cx - D
    con[n:2*n] = 1.0 - cx - D
    con[2*n:3*n] = cy - D
    con[3*n:4*n] = 1.0 - cy - D
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    con[4*n:] = dx**2 + dy**2 - 4.0 * D**2
    return con

def objective_func(vars_arr, n):
    return -vars_arr[2*n]

def solve_lp_radii(centers, n):
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = max(1e-9, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-7):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def generate_hex_configs(n, rng):
    configs = []
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 6, 4],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 4, 5],
        [5, 5, 5, 5, 6], [7, 6, 6, 7], [6, 7, 6, 7], [5, 6, 5, 6, 5, 1]
    ]
    
    for pat in row_patterns:
        if sum(pat) < n: continue
        pts = []
        r0 = 0.095
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        while len(pts) < n:
            pts.append([0.5, 0.5])
            
        base = np.array(pts[:n])
        configs.append(base)
        
        for _ in range(4):
            p = base + rng.uniform(-0.025, 0.025, (n, 2))
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(10):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    return configs

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    configs = generate_hex_configs(n, rng)
    
    best_D = 0.0
    best_centers_equal = None
    
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.2)]
    
    # Phase 1: Maximize equal radius D
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP', 
                           bounds=bounds_opt, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if np.isfinite(res.fun):
                D_opt = res.x[-1]
                if D_opt > best_D:
                    best_D = D_opt
                    best_centers_equal = res.x[:2*n].reshape(n, 2).copy()
        except Exception:
            continue
            
    current_best_centers = best_centers_equal
    current_best_sum = -1.0
    current_best_radii = None
    
    # Phase 2: LP refinement on best equal-radius centers
    r_lp = solve_lp_radii(current_best_centers, n)
    if r_lp is not None:
        current_best_sum = np.sum(r_lp)
        current_best_radii = r_lp
        
    # Phase 3: Local center perturbation to squeeze out more sum
    for step in np.linspace(0.02, 0.001, 15):
        improved = False
        for _ in range(40):
            idx = rng.integers(n)
            old_pos = current_best_centers[idx].copy()
            current_best_centers[idx] += rng.uniform(-step, step, 2)
            current_best_centers[idx] = np.clip(current_best_centers[idx], 0.01, 0.99)
            
            r_test = solve_lp_radii(current_best_centers, n)
            if r_test is not None:
                s_test = np.sum(r_test)
                if s_test > current_best_sum + 1e-9:
                    current_best_sum = s_test
                    current_best_radii = r_test.copy()
                    improved = True
                else:
                    current_best_centers[idx] = old_pos
            else:
                current_best_centers[idx] = old_pos
        if not improved:
            break
            
    best_centers = current_best_centers
    best_radii = current_best_radii
    
    # Safety scaling to guarantee strict numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
