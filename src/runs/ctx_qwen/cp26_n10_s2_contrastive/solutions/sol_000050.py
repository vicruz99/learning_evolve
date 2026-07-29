# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000024 (state 7d29769f) state=5c8f47d4 sum of radii=2.626740 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIRS_I)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary and non-overlap (squared)."""
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

def get_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((NUM_PAIRS, n))
    b_ub = np.zeros(NUM_PAIRS)
    
    for k in range(NUM_PAIRS):
        i, j = PAIRS_I[k], PAIRS_J[k]
        d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = d
        
    bounds = []
    for i in range(n):
        mx, my = centers[i]
        ub = min(mx, 1.0-mx, my, 1.0-my)
        bounds.append((0.0, max(0.0, ub)))
        
    for method in ['highs', 'interior-point', 'revised simplex']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return res.x
        except Exception:
            continue
    return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate structured hexagonal initializations
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6], [8, 6, 5, 4, 3], [6, 6, 6, 6, 2], [5, 5, 5, 5, 6]
    ]
    inits = []
    for pat in patterns:
        pts = []
        y = 0.05
        dy = 0.165
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.085
            x = 0.05 + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 0.17
            y += dy
        while len(pts) < N:
            pts.append([0.5, 0.5])
        inits.append(np.array(pts[:N]))
        
    # Add random initializations
    for seed in range(10):
        rng = np.random.default_rng(seed*7+3)
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))

    trial = 0
    max_trials = 80
    
    for base in inits:
        for nl in [0.0, 0.01, 0.03, 0.05]:
            if trial >= max_trials:
                break
                
            rng = np.random.default_rng(trial*17+5)
            c_init = base + rng.normal(0, nl, base.shape)
            c_init = np.clip(c_init, 0.02, 0.98)
            
            x0 = np.zeros(3*N)
            x0[0::3] = c_init[:, 0]
            x0[1::3] = c_init[:, 1]
            x0[2::3] = 0.03 
            
            curr_x = x0.copy()
            
            # Block coordinate descent: alternate SLSQP and LP
            for _ in range(2):
                try:
                    res = minimize(objective, curr_x, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                    if res.success:
                        curr_x = res.x
                except Exception:
                    break
                    
                curr_c = np.column_stack((curr_x[0::3], curr_x[1::3]))
                curr_r = get_lp_radii(curr_c)
                curr_x[2::3] = curr_r
                
            # Final refinement pass
            try:
                res = minimize(objective, curr_x, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                    curr_r = get_lp_radii(curr_c)
                    curr_s = np.sum(curr_r)
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
            except Exception:
                pass
                
            trial += 1

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = get_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Strict numerical validation and adjustment
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    for i in range(N):
        mx, my = c_final[i]
        r_final[i] = min(r_final[i], mx, 1.0-mx, my, 1.0-my, 0.5)
        r_final[i] = max(0.0, r_final[i])
        
    for _ in range(20):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = PAIRS_I[k], PAIRS_J[k]
            d = np.sqrt((c_final[i,0]-c_final[j,0])**2 + (c_final[i,1]-c_final[j,1])**2)
            if d < r_final[i] + r_final[j] - 1e-12:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
