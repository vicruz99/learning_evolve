# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=4004305a sum of radii=2.621593 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """
    Solves an LP to find optimal radii for fixed centers.
    Maximizes sum(r) subject to r_i + r_j <= dist(i,j) and boundary limits.
    """
    n = centers.shape[0]
    # Boundary limits for each circle
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 0.0)

    c_obj = -np.ones(n)  # Maximize sum(r) => minimize -sum(r)
    bounds = [(0.0, lims[i]) for i in range(n)]

    # Pairwise distance constraints
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)

    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    b_ub = dists[idx_i, idx_j]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return res.x
    except Exception:
        pass
    return None

def objective(vars, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars[2 * n:])

def constraints(vars, n):
    """
    Computes inequality constraints >= 0 for valid packing.
    1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    2. Non-overlap: ||c_i - c_j||^2 - (r_i + r_j)^2 >= 0
    """
    cx = vars[:n]
    cy = vars[n:2 * n]
    r = vars[2 * n:]
    
    c = []
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend((d2 - rs2)[mask])
    
    return np.array(c)

def get_hex_init(r0, n):
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
            x += 2 * r0
        y += dy
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    starts = []
    
    # 1. Hexagonal lattices with different base radii
    for r0 in [0.09, 0.10, 0.105, 0.11]:
        starts.append(get_hex_init(r0, n))
        starts.append(get_hex_init(r0, n) + np.random.uniform(-0.02, 0.02, (n, 2)))
        
    # 2. Random dense starts
    for _ in range(8):
        starts.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    # Upper triangle mask for fast validation
    overlap_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    def try_optimize(cfg):
        nonlocal best_sum, best_centers, best_radii
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.085  # Strictly feasible initial radius
        
        try:
            res = minimize(
                objective, x0, args=(n,), method='SLSQP', 
                bounds=bounds_opt, constraints=cons_dict,
                options={'maxiter': 12000, 'ftol': 1e-14}
            )
            
            cx = res.x[:n]
            cy = res.x[n:2 * n]
            centers = np.column_stack((cx, cy))
            radii = np.maximum(res.x[2 * n:], 1e-9)
            
            # LP refinement for radii given these centers
            r_lp = solve_radii_lp(centers)
            if r_lp is not None:
                radii = r_lp
                
            # Validation
            valid = True
            if np.any(centers[:, 0] < radii - 1e-12) or np.any(centers[:, 0] > 1 - radii + 1e-12) or \
               np.any(centers[:, 1] < radii - 1e-12) or np.any(centers[:, 1] > 1 - radii + 1e-12):
                valid = False
            else:
                cx_m = centers[:, 0][:, None] - centers[:, 0][None, :]
                cy_m = centers[:, 1][:, None] - centers[:, 1][None, :]
                d2 = cx_m**2 + cy_m**2
                rs = radii[:, None] + radii[None, :]
                if np.any(d2[overlap_mask] < rs[overlap_mask]**2 - 1e-11):
                    valid = False
                    
            if valid:
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
        except Exception:
            pass

    # Run from all initial configurations
    for cfg in starts:
        try_optimize(cfg)
        
    # Local search refinement on best result
    if best_centers is not None:
        for _ in range(6):
            pert = best_centers + np.random.uniform(-0.004, 0.004, (n, 2))
            pert = np.clip(pert, 0.02, 0.98)
            try_optimize(pert)
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = get_hex_init(0.095, n)
        best_radii = np.full(n, 0.095)
        best_sum = np.sum(best_radii)
        
    # Final safety scaling to guarantee strict validity against checker tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
