# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=73db2b13 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((len(I_IDX), n))
    A_ub[np.arange(len(I_IDX)), I_IDX] = 1.0
    A_ub[np.arange(len(I_IDX)), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.zeros(n)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary and non-overlap (squared for smoothness)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    # Pairwise non-overlap constraints
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    r_sum = r[I_IDX] + r[J_IDX]
    c = np.concatenate([c, dx**2 + dy**2 - r_sum**2])
    return c

BOUNDS_OPT = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
CONS_OPT = {'type': 'ineq', 'fun': constraints}

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    best_sum = 0.0
    best_c = None
    best_r = None
    rng = np.random.default_rng(2024)
    
    # Helper to execute optimization and track best
    def run_opt(x0, iters=10000):
        nonlocal best_sum, best_c, best_r
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=BOUNDS_OPT,
                           constraints=CONS_OPT, options={'maxiter': iters, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            curr_c = np.column_stack((cx, cy))
            curr_r = solve_lp_radii(curr_c)
            curr_s = np.sum(curr_r)
            
            if curr_s > best_sum:
                best_sum = curr_s
                best_c = curr_c.copy()
                best_r = curr_r.copy()
        except Exception:
            pass

    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice variations
    for _ in range(25):
        c = np.zeros((N, 2))
        idx = 0
        y = 0.05 + rng.uniform(0, 0.05)
        row = 0
        sp = 0.13 + rng.uniform(0, 0.07)
        while idx < N and y < 0.95:
            x = 0.05 + (row % 2) * sp / 2.0 + rng.uniform(-0.01, 0.01)
            while x < 0.95 and idx < N:
                c[idx] = [x, y]
                x += sp
                idx += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.06, 0.94)
        r_init = np.full(N, 0.035)
        x0 = np.zeros(3 * N)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = r_init
        inits.append(x0)
        
    # 2. Grid lattice variations
    for _ in range(15):
        c = np.zeros((N, 2))
        idx = 0
        step = 0.16 + rng.uniform(-0.02, 0.03)
        y = 0.05 + rng.uniform(0, 0.05)
        while y < 0.95 and idx < N:
            x = 0.05 + rng.uniform(-0.01, 0.01)
            while x < 0.95 and idx < N:
                c[idx] = [x, y]
                x += step
                idx += 1
            y += step
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.06, 0.94)
        r_init = np.full(N, 0.035)
        x0 = np.zeros(3 * N)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = r_init
        inits.append(x0)
        
    # 3. Random uniform placements
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c += rng.normal(0, 0.03, c.shape)
        c = np.clip(c, 0.08, 0.92)
        r_init = np.full(N, 0.03)
        x0 = np.zeros(3 * N)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = r_init
        inits.append(x0)
        
    # Stage 1: Broad search from diverse starts
    for x0 in inits:
        run_opt(x0, iters=6000)
        
    # Stage 2: Local refinement & basin hopping
    if best_c is not None:
        # Controlled perturbations around best solution
        for _ in range(35):
            scale = rng.uniform(0.002, 0.012)
            c_pert = best_c + rng.normal(0, scale, best_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert = solve_lp_radii(c_pert) * 0.98
            x_pert = np.zeros(3 * N)
            x_pert[0::3] = c_pert[:, 0]
            x_pert[1::3] = c_pert[:, 1]
            x_pert[2::3] = np.maximum(r_pert, 1e-5)
            run_opt(x_pert, iters=10000)
            
        # Iterative LP-SLSQP refinement cycle
        for _ in range(20):
            c_cur = best_c + rng.normal(0, 0.004, best_c.shape)
            c_cur = np.clip(c_cur, 0.03, 0.97)
            r_cur = solve_lp_radii(c_cur)
            x_start = np.zeros(3 * N)
            x_start[0::3] = c_cur[:, 0]
            x_start[1::3] = c_cur[:, 1]
            x_start[2::3] = np.maximum(r_cur, 1e-5)
            run_opt(x_start, iters=12000)
            
    # Fallback safety net
    if best_c is None:
        best_c = np.random.uniform(0.2, 0.8, (N, 2))
        best_r = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    # Stage 3: Strict post-processing to guarantee validity
    c_final = best_c.copy()
    r_final = best_r.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(r_final[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(len(I_IDX)):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
            if d < r_final[i] + r_final[j] - 1e-10:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    best_sum = float(np.sum(r_final))
    return c_final, r_final, best_sum
