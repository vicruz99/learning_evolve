# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000126 (state 8609ace4) state=4b9aa38e sum of radii=2.618066 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds = [(0.0, None)] * n
    
    num_boundary = 4 * n
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_boundary + num_pairs, n))
    b_ub = np.zeros(num_boundary + num_pairs)
    
    idx = 0
    # Boundary constraints: r_i <= dist to each wall
    for i in range(n):
        x, y = centers[i]
        for lim in (x, 1.0 - x, y, 1.0 - y):
            A_ub[idx, i] = 1.0
            b_ub[idx] = lim
            idx += 1
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def obj_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def cons_func(x, n):
    """Computes inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2 * n]
    r = x[2 * n:]
    
    c = np.empty(4 * n + n * (n - 1) // 2)
    c[:n] = cx - r
    c[n:2*n] = 1.0 - cx - r
    c[2*n:3*n] = cy - r
    c[3*n:4*n] = 1.0 - cy - r
    
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c[4*n:] = (cx_m**2 + cy_m**2 - r_m**2)[mask]
    
    return c

def get_hex_init(n, r0, rng):
    """Generates an initial hexagonal grid of n circles with radius r0."""
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    while len(pts) < n and y + r0 < 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += dy
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    
    inits = []
    # Structured hexagonal starts with varying densities
    for r0 in [0.08, 0.085, 0.09, 0.095, 0.10]:
        base = get_hex_init(n, r0, rng)
        inits.append(base)
        for _ in range(3):
            pert = base + rng.uniform(-0.015, 0.015, base.shape)
            inits.append(np.clip(pert, 0.02, 0.98))
            
    # Random starts
    for _ in range(5):
        inits.append(np.clip(rng.uniform(0.1, 0.9, (n, 2)), 0.02, 0.98))
        
    best_sum = -1.0
    best_c = None
    best_r = None
    
    cons_dict = {'type': 'ineq', 'fun': cons_func, 'args': (n,)}
    
    for cfg in inits:
        # Compute strictly feasible initial radii
        dists = np.sqrt(np.sum((cfg[:, None] - cfg[None, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists, axis=1) / 2.0
        wall = np.minimum(np.minimum(cfg[:, 0], 1.0 - cfg[:, 0]), np.minimum(cfg[:, 1], 1.0 - cfg[:, 1]))
        init_r = np.minimum(min_pair, wall) * 0.85
        
        v0 = np.zeros(3 * n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = init_r
        
        try:
            res = minimize(obj_func, v0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            cx = res.x[:n]
            cy = res.x[n:2*n]
            rs = res.x[2*n:]
            
            centers = np.column_stack((cx, cy))
            
            # Quick feasibility check before LP
            if np.any(rs < 1e-7) or np.any(cx < rs - 1e-6) or np.any(cx > 1.0 - rs + 1e-6) or \
               np.any(cy < rs - 1e-6) or np.any(cy > 1.0 - rs + 1e-6):
                continue
                
            # LP refinement extracts maximum possible radii for fixed centers
            lp_r, lp_sum = solve_lp_radii(centers)
            if lp_r is not None and lp_sum > best_sum:
                best_sum = lp_sum
                best_c = centers.copy()
                best_r = lp_r.copy()
        except Exception:
            continue
            
    if best_c is None:
        best_c = inits[0]
        best_r = np.full(n, 0.08)
        best_sum = np.sum(best_r)
        
    # Safety scaling to guarantee strict numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs_sum = best_r[i] + best_r[j]
            if rs_sum > 1e-12:
                scale = min(scale, d / rs_sum)
                
    best_r *= scale * 0.999999
    best_sum = np.sum(best_r)
    
    return best_c, best_r, float(best_sum)
