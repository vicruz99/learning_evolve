# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000036 (state d4cf115e) state=07dbf4af sum of radii=1.553569 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to find radii that maximize sum(radii)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    dx = centers[PAIRS_I, 0] - centers[PAIRS_J, 0]
    dy = centers[PAIRS_I, 1] - centers[PAIRS_J, 1]
    dists = np.hypot(dx, dy)
    b_ub[:] = dists
    
    A_ub[np.arange(NUM_PAIRS), PAIRS_I] = 1.0
    A_ub[np.arange(NUM_PAIRS), PAIRS_J] = 1.0
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(0.0, ub)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0), -res.fun
        except Exception:
            pass
    return np.zeros(n), 0.0

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (squared)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[PAIRS_I] - cx[PAIRS_J]
    dy = cy[PAIRS_I] - cy[PAIRS_J]
    r_sum = r[PAIRS_I] + r[PAIRS_J]
    c[4*N:] = dx**2 + dy**2 - r_sum**2
    return c

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Hexagonal patterns with varying spacing
    for seed in range(25):
        srng = np.random.RandomState(seed)
        sp = 0.15 + srng.uniform(0.0, 0.05)
        c = np.zeros((N, 2))
        idx = 0
        row = 0
        y = sp/2
        while idx < N and y < 1.0 - sp/2:
            x_off = sp/2 + (row % 2) * sp/2
            col = 0
            while x_off + col*sp < 1.0 - sp/2 and idx < N:
                c[idx] = [x_off + col*sp, y]
                idx += 1
                col += 1
            y += sp * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = srng.uniform(0.1, 0.9, 2)
            idx += 1
        c += srng.normal(0, 0.008, c.shape)
        c = np.clip(c, 0.02, 0.98)
        r, _ = solve_lp_radii(c)
        inits.append((c, r*0.96))
        
    # 2. Grid patterns
    for seed in range(15):
        srng = np.random.RandomState(seed + 100)
        step = 0.16 + srng.uniform(-0.01, 0.02)
        c = np.zeros((N, 2))
        idx = 0
        y = step/2
        while y < 1.0 - step/2 and idx < N:
            x = step/2
            while x < 1.0 - step/2 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += step
            y += step
        while idx < N:
            c[idx] = srng.uniform(0.1, 0.9, 2)
            idx += 1
        c += srng.normal(0, 0.008, c.shape)
        c = np.clip(c, 0.02, 0.98)
        r, _ = solve_lp_radii(c)
        inits.append((c, r*0.96))

    # 3. Random feasible starts
    for seed in range(20):
        srng = np.random.RandomState(seed + 200)
        c = srng.uniform(0.1, 0.9, (N, 2))
        r, _ = solve_lp_radii(c)
        inits.append((c, r*0.96))

    # Main optimization loop
    for c0, r0 in inits:
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            co = res.x[0::3]
            cy = res.x[1::3]
            ro = np.maximum(res.x[2::3], 0.0)
            curr_c = np.column_stack((co, cy))
            
            r_lp, s_lp = solve_lp_radii(curr_c)
            if s_lp > best_sum:
                best_sum = s_lp
                best_c = curr_c.copy()
                best_r = r_lp.copy()
        except Exception:
            pass
            
    # Local refinement around best solution
    if best_c is not None:
        for trial in range(40):
            srng = np.random.RandomState(trial * 31 + 7)
            pert_scale = 0.01 * (0.85 ** (trial // 10))
            c_p = best_c + srng.normal(0, pert_scale, best_c.shape)
            c_p = np.clip(c_p, 0.02, 0.98)
            r_p, _ = solve_lp_radii(c_p)
            
            x0 = np.zeros(3*N)
            x0[0::3] = c_p[:, 0]
            x0[1::3] = c_p[:, 1]
            x0[2::3] = r_p * 0.96
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                co = res.x[0::3]
                cy = res.x[1::3]
                ro = np.maximum(res.x[2::3], 0.0)
                curr_c = np.column_stack((co, cy))
                
                r_lp, s_lp = solve_lp_radii(curr_c)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = curr_c.copy()
                    best_r = r_lp.copy()
            except Exception:
                pass
                
    # Strict post-processing to guarantee validity
    if best_c is not None:
        c_final = best_c.copy()
        r_final = best_r.copy()
        
        # Enforce boundary constraints strictly
        for i in range(N):
            mx = min(c_final[i,0], 1.0-c_final[i,0], c_final[i,1], 1.0-c_final[i,1])
            r_final[i] = min(r_final[i], max(0.0, mx - 1e-9))
            
        # Iteratively resolve any remaining numerical overlaps
        for _ in range(100):
            changed = False
            for k in range(NUM_PAIRS):
                i, j = PAIRS_I[k], PAIRS_J[k]
                d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    exc = r_final[i] + r_final[j] - d
                    r_final[i] -= exc * 0.5
                    r_final[j] -= exc * 0.5
                    changed = True
            if not changed:
                break
        r_final = np.maximum(r_final, 0.0)
        best_sum = np.sum(r_final)
        
    # Fallback safety net
    if best_c is None:
        best_c = np.random.uniform(0.2, 0.8, (N, 2))
        best_r, _ = solve_lp_radii(best_c)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, float(best_sum)
