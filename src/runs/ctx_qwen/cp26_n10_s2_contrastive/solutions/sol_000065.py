# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000038 (state cf517c54) state=6bddc338 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """
    Given fixed centers, solve an LP to find radii that maximize the sum of radii
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    A_ub = np.zeros((n_pairs, n))
    A_ub[np.arange(n_pairs), i_idx] = 1.0
    A_ub[np.arange(n_pairs), j_idx] = 1.0
    
    dx = centers[i_idx, 0] - centers[j_idx, 0]
    dy = centers[i_idx, 1] - centers[j_idx, 1]
    b_ub = np.sqrt(dx**2 + dy**2)
    
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    except Exception:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='interior-point')
        except Exception:
            return np.zeros(n), 0.0
            
    if res.success:
        radii = np.maximum(res.x, 0.0)
        return radii, -res.fun
    return np.zeros(n), 0.0

def obj_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def constr_func(x, n):
    """
    Inequality constraints: boundary and pairwise non-overlap.
    Formulated with squared distances for numerical stability.
    """
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    n_cons = 4*n + n*(n-1)//2
    c = np.empty(n_cons)
    
    # Boundary constraints
    c[:n] = cx - r
    c[n:2*n] = 1.0 - cx - r
    c[2*n:3*n] = cy - r
    c[3*n:4*n] = 1.0 - cy - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = cx[i_idx] - cx[j_idx]
    dy = cy[i_idx] - cy[j_idx]
    dist_sq = dx**2 + dy**2
    r_sum = r[i_idx] + r[j_idx]
    c[4*n:] = dist_sq - r_sum**2
    
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constr_func, 'args': (n,)}
    
    best_sum = 0.0
    best_x = None
    
    inits = []
    
    # 1. Hexagonal lattice
    r_h = 0.09
    pts = []
    y = r_h
    row = 0
    while len(pts) < n:
        x = r_h + (row % 2) * r_h
        while x < 1 - r_h and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_h
        y += r_h * np.sqrt(3)
        row += 1
    pts = np.array(pts[:n])
    inits.append(pts)
    
    # 2. 5x5 grid + 1 center
    grid = []
    for i in range(5):
        for j in range(5):
            grid.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    grid.append([0.5, 0.5])
    grid = np.array(grid[:n])
    inits.append(grid)
    
    # 3. Random placements
    for s in range(10):
        rng = np.random.RandomState(s * 13 + 1)
        p = rng.uniform(0.15, 0.85, (n, 2))
        inits.append(p)
        
    # 4. Perturbed hexagonal lattices
    for s in range(15):
        rng = np.random.RandomState(s * 17 + 2)
        p = pts + rng.normal(0, 0.025, pts.shape)
        p = np.clip(p, 0.05, 0.95)
        inits.append(p)
        
    # Main optimization loop
    for base_pts in inits:
        lp_r, _ = solve_radii_lp(base_pts)
        x0 = np.concatenate([base_pts[:, 0], base_pts[:, 1], lp_r])
        x0[2*n:] = np.maximum(x0[2*n:], 1e-5)
        
        try:
            res = minimize(obj_func, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Local perturbation refinement
    if best_x is not None:
        for _ in range(30):
            rng = np.random.RandomState(_ * 19 + 5)
            xp = best_x + rng.normal(0, 0.003, len(best_x))
            xp = np.clip(xp, [b[0] for b in bounds], [b[1] for b in bounds])
            xp[2*n:] = np.maximum(xp[2*n:], 1e-5)
            
            try:
                res = minimize(obj_func, xp, args=(n,), method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
                
    if best_x is None:
        best_x = np.concatenate([inits[0][:, 0], inits[0][:, 1], np.full(n, 0.08)])
        
    cx = best_x[:n]
    cy = best_x[n:2*n]
    centers = np.column_stack((cx, cy))
    
    # Final LP refinement to extract maximum radii for optimized centers
    radii, _ = solve_radii_lp(centers)
    
    # Post-processing: enforce strict validity within numerical tolerance
    for i in range(n):
        mx = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if radii[i] > mx + 1e-12:
            radii[i] = max(0.0, mx - 1e-9)
            
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < radii[i] + radii[j] - 1e-12:
                    ex = radii[i] + radii[j] - d
                    radii[i] -= ex / 2.0
                    radii[j] -= ex / 2.0
                    changed = True
        if not changed:
            break
            
    final_sum = np.sum(radii)
    return centers, radii, float(final_sum)
