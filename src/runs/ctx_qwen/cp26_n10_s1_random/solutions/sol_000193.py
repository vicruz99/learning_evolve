# sol_000193 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000148 (state 41e5ee41) state=2527644c sum of radii=2.630169 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_constraints(vars_arr, n):
    """Computes inequality constraints >= 0 for valid packing."""
    x = vars_arr[0::3]
    y = vars_arr[1::3]
    r = vars_arr[2::3]
    c = []
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    idx = np.triu_indices(n, k=1)
    c.append(dists[idx] - r_sum[idx])
    return np.concatenate(c)

def solve_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        x, y = centers[i]
        lims = (x, 1-x, y, 1-y)
        for lim in lims:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 0.0

def objective_func(vars_arr):
    """Objective: minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars_arr[2::3])

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n,)}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = []
    
    # Hexagonal patterns with various row distributions
    row_patterns = [
        [5, 5, 5, 5, 6], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], 
        [4, 6, 6, 6, 4], [5, 5, 5, 5, 5, 1], [6, 6, 5, 5, 4]
    ]
    
    for r0 in [0.085, 0.09, 0.095, 0.10]:
        for pattern in row_patterns:
            pts = []
            y = r0
            row_idx = 0
            for count in pattern:
                shift = r0 if row_idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(count):
                    if len(pts) >= n:
                        break
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3) * r0
                row_idx += 1
            if len(pts) >= n:
                inits.append(np.array(pts[:n]))
                
    # Regular grid + center
    gx = np.linspace(0.15, 0.85, 5)
    gy = np.linspace(0.15, 0.85, 5)
    pts = np.array([(x, y) for y in gy for x in gx])
    pts = np.vstack([pts, [0.5, 0.5]])
    inits.append(pts)
    
    # Random placements
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: SLSQP from diverse starts
    for cfg in inits:
        for _ in range(4):
            pert = cfg.copy()
            pert += rng.uniform(-0.02, 0.02, (n, 2))
            pert = np.clip(pert, 0.05, 0.95)
            
            x0 = np.zeros(3*n)
            x0[0::3] = pert[:, 0]
            x0[1::3] = pert[:, 1]
            x0[2::3] = 0.06  # Feasible start
            
            try:
                res = minimize(objective_func, x0, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-14})
                if np.isfinite(res.fun):
                    c_s = np.column_stack((res.x[0::3], res.x[1::3]))
                    # Phase 2: Exact LP refinement on fixed centers
                    r_lp, s_lp = solve_lp(c_s)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_s.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass
                
    # Phase 3: Stochastic hill climbing on centers using LP evaluation
    if best_centers is not None:
        curr_centers = best_centers.copy()
        curr_radii, curr_sum = solve_lp(curr_centers)
        best_sum = curr_sum
        best_radii = curr_radii
        best_centers = curr_centers
        
        for step in range(2000):
            i = rng.integers(n)
            old_c = curr_centers[i].copy()
            step_size = 0.025 * (0.998 ** step)
            curr_centers[i] += rng.uniform(-step_size, step_size, 2)
            curr_centers[i] = np.clip(curr_centers[i], 0.02, 0.98)
            
            r_new, s_new = solve_lp(curr_centers)
            if s_new > curr_sum:
                curr_sum = s_new
                curr_radii = r_new
                best_sum = curr_sum
                best_radii = curr_radii.copy()
                best_centers = curr_centers.copy()
            else:
                curr_centers[i] = old_c
                
    # Phase 4: Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-9:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((best_centers[i,0]-best_centers[j,0])**2 + (best_centers[i,1]-best_centers[j,1])**2)
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d/rs)
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
