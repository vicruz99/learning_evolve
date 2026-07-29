# sol_000117 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=ebe020d5 sum of radii=2.581358 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0)
        except Exception:
            continue
    return np.full(n, 0.01)

def objective(params):
    """Minimize negative sum of radii."""
    return -np.sum(params[2::3])

def constraints(params):
    """Inequality constraints: boundary and non-overlap (must be >= 0)."""
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with varying spacing
    for sp in np.linspace(0.13, 0.22, 10):
        pts = []
        y = sp / 2.0
        row = 0
        while len(pts) < N and y < 1.0 - sp / 2.0:
            x_start = sp / 2.0 + (row % 2) * sp / 2.0
            col = 0
            while x_start + col * sp < 1.0 - sp / 2.0 and len(pts) < N:
                pts.append([x_start + col * sp, y])
                col += 1
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while len(pts) < N:
            pts.append(rng.uniform(0.2, 0.8, 2))
        inits.append(np.array(pts[:N]))
        
    # 2. Corner/Edge focused patterns
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    edges = [[0.5, 0.05], [0.95, 0.5], [0.5, 0.95], [0.05, 0.5]]
    mid = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
    base_pts = corners + edges + mid
    while len(base_pts) < N:
        base_pts.append(rng.uniform(0.3, 0.7, 2))
    inits.append(np.array(base_pts[:N]))
    
    # 3. Random uniform
    for _ in range(25):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # Phase 2: Multi-start SLSQP with LP initialization
    for base in inits:
        c0 = np.clip(base + rng.normal(0, 0.005, base.shape), 0.02, 0.98)
        r0 = solve_radii_lp(c0) * 0.96
        r0 = np.maximum(r0, 1e-5)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, 
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                co = np.column_stack((res.x[0::3], res.x[1::3]))
                ro = solve_radii_lp(co)
                so = np.sum(ro)
                if so > best_sum:
                    best_sum = so
                    best_centers = co.copy()
                    best_radii = ro.copy()
        except Exception:
            continue
            
    # Phase 3: Basin Hopping & Perturbation Refinement
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(120):
            # Adaptive noise schedule
            scale = 0.002 * (1.0 + 0.6 * np.random.rand()) * (1.0 + 0.3 * (step < 30))
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_radii_lp(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
                # Local SLSQP polish after successful jump
                x0 = np.zeros(3 * N)
                x0[0::3] = curr_c[:, 0]
                x0[1::3] = curr_c[:, 1]
                x0[2::3] = curr_r * 0.98
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        co = np.column_stack((res.x[0::3], res.x[1::3]))
                        ro = solve_radii_lp(co)
                        so = np.sum(ro)
                        if so > curr_s:
                            curr_c, curr_r, curr_s = co, ro, so
                except Exception:
                    pass
                    
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
            # Radius inflation trick to escape tight contact local minima
            if step % 12 == 0 and step > 0:
                inflated_r = curr_r * 1.06
                x0 = np.zeros(3 * N)
                x0[0::3] = curr_c[:, 0]
                x0[1::3] = curr_c[:, 1]
                x0[2::3] = inflated_r
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        co = np.column_stack((res.x[0::3], res.x[1::3]))
                        ro = solve_radii_lp(co)
                        so = np.sum(ro)
                        if so > best_sum:
                            best_sum = so
                            best_centers = co.copy()
                            best_radii = ro.copy()
                            curr_c, curr_r, curr_s = co, ro, so
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, (N, 2))
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validity within validator tolerance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))
