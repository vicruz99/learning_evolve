# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000052 (state 0d4d18bd) state=eb2c8228 sum of radii=2.625974 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    k = 0
    for i in range(n):
        v = max(1e-9, centers[i, 0]); A_ub[k, i] = 1.0; b_ub[k] = v; k += 1
        v = max(1e-9, 1.0 - centers[i, 0]); A_ub[k, i] = 1.0; b_ub[k] = v; k += 1
        v = max(1e-9, centers[i, 1]); A_ub[k, i] = 1.0; b_ub[k] = v; k += 1
        v = max(1e-9, 1.0 - centers[i, 1]); A_ub[k, i] = 1.0; b_ub[k] = v; k += 1
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[k, i] = 1.0; A_ub[k, j] = 1.0; b_ub[k] = max(1e-9, d); k += 1
    bounds = [(0.0, None)] * n
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.x if res.success else np.full(n, 0.01)

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Returns vector of inequality constraints g(vars) >= 0."""
    n = N_CIRCLES
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Overlap constraints (squared distance form for better differentiability)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    i_idx, j_idx = np.tril_indices(n, -1)
    c = np.concatenate([c, dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2 - dr[i_idx, j_idx]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N_CIRCLES):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_init_centers(strategy='hex', seed=0):
    """Generates initial center configurations."""
    np.random.seed(seed)
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    if strategy == 'hex':
        r = 0.09
        y = r
        row = 0
        idx = 0
        while idx < n and y + r <= 1.0:
            x = r if row % 2 == 0 else 2 * r
            while idx < n and x + r <= 1.0:
                centers[idx] = [x, y]
                idx += 1
                x += 2 * r
            y += np.sqrt(3) * r
            row += 1
        while idx < n:
            centers[idx] = np.random.uniform(0.1, 0.9, 2)
            idx += 1
    elif strategy == 'grid':
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < n:
                    centers[idx] = [0.1 + j * 0.18, 0.1 + i * 0.16]
                    idx += 1
    elif strategy == 'rand':
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        
    centers += np.random.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def hill_climb(centers, steps=150):
    """Coordinate ascent on centers using LP to evaluate radius sum."""
    n = centers.shape[0]
    centers = centers.copy()
    radii = solve_lp_radii(centers)
    best_sum = np.sum(radii)
    step = 0.04
    
    for t in range(steps):
        step = 0.04 * (0.997 ** t)
        idx = np.random.randint(n)
        old = centers[idx].copy()
        
        centers[idx] += np.random.normal(0, step, 2)
        centers[idx] = np.clip(centers[idx], 0.01, 0.99)
        
        radii = solve_lp_radii(centers)
        curr_sum = np.sum(radii)
        if curr_sum > best_sum:
            best_sum = curr_sum
        else:
            centers[idx] = old
            
    return centers, solve_lp_radii(centers), best_sum

def run_packing():
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_vars = None
    bounds = get_bounds()
    
    # Phase 1: Generate diverse starts and refine centers via hill-climbing
    inits = []
    for s in range(4):
        for strat in ['hex', 'grid', 'rand']:
            c = generate_init_centers(strat, seed=s * 3 + 1)
            c, r, s_val = hill_climb(c, steps=120)
            inits.append((c, r, s_val))
            
    # Phase 2: SLSQP joint optimization on all candidates
    for c_init, r_init, _ in inits:
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 3: Perturbation & Refinement to escape local minima
    if best_vars is not None:
        for _ in range(4):
            x_pert = best_vars + np.random.normal(0, 0.001, best_vars.shape)
            r_p = np.maximum(x_pert[2::3], 0.005)
            x_pert[0::3] = np.clip(x_pert[0::3], r_p, 1.0 - r_p)
            x_pert[1::3] = np.clip(x_pert[1::3], r_p, 1.0 - r_p)
            x_pert[2::3] = r_p
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraints},
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun) and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_vars = res.x.copy()
            except Exception:
                pass
                
    # Fallback
    if best_vars is None:
        c, r, _ = inits[0]
        best_vars = np.zeros(3 * N_CIRCLES)
        best_vars[0::3] = c[:, 0]
        best_vars[1::3] = c[:, 1]
        best_vars[2::3] = r
        best_sum = np.sum(r)
        
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    
    # Phase 4: Final LP polish to squeeze out maximum radii for optimized centers
    radii_lp = solve_lp_radii(centers)
    if np.sum(radii_lp) > np.sum(radii):
        radii = radii_lp
        best_sum = np.sum(radii)
        
    # Phase 5: Strict validity repair
    for _ in range(50):
        valid = True
        for i in range(N_CIRCLES):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1 - radii[i] + 1e-9 or \
               centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1 - radii[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        radii *= 0.9999
        
    return centers, radii, float(np.sum(radii))
