# sol_000177 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=0ce77dda sum of radii=2.627680 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds = [(0, None)] * n
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(lim)
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def joint_obj(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def joint_cons(x, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2 * n]
    r = x[2 * n:]
    
    con = []
    # Boundary constraints
    con.append(cx - r)
    con.append(1.0 - cx - r)
    con.append(cy - r)
    con.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    idx_i, idx_j = np.triu_indices(n, k=1)
    dist_sq = cx_m[idx_i, idx_j] ** 2 + cy_m[idx_i, idx_j] ** 2
    r_sum_sq = r_m[idx_i, idx_j] ** 2
    con.append(dist_sq - r_sum_sq)
    
    return np.concatenate(con)

def make_hex(rows_pattern, r0=0.09):
    """Generates initial positions on a hexagonal lattice."""
    n = 26
    pts = []
    y = r0
    for ri, cnt in enumerate(rows_pattern):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Structured patterns summing to 26
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [6, 6, 6, 4, 4], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6], [5, 6, 6, 5, 4]
    ]
    
    inits = []
    for p in patterns:
        if sum(p) >= n:
            pts = make_hex(p)
            inits.append(pts)
            inits.append(np.clip(pts + rng.uniform(-0.02, 0.02, pts.shape), 0.05, 0.95))
            
    # Random starts to explore diverse basins
    for _ in range(25):
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    def try_opt(cfg):
        nonlocal best_sum, best_centers, best_radii
        
        # Phase 1: Get optimal radii for fixed centers via LP
        r_lp = solve_lp_radii(cfg)
        if r_lp is None:
            return
            
        s_lp = np.sum(r_lp)
        if s_lp > best_sum:
            best_sum = s_lp
            best_centers = cfg.copy()
            best_radii = r_lp.copy()
            
        # Phase 2: Joint SLSQP refinement
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = r_lp
        
        try:
            res = minimize(joint_obj, x0, args=(n,), method='SLSQP',
                           bounds=bounds_vars,
                           constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2 * n]
                r = np.maximum(res.x[2 * n:], 1e-9)
                centers_opt = np.column_stack((cx, cy))
                radii_opt = r
                
                # Strict validity verification
                valid = True
                if np.any(centers_opt[:, 0] < radii_opt - 1e-9) or np.any(centers_opt[:, 0] > 1 - radii_opt + 1e-9):
                    valid = False
                if np.any(centers_opt[:, 1] < radii_opt - 1e-9) or np.any(centers_opt[:, 1] > 1 - radii_opt + 1e-9):
                    valid = False
                
                if valid:
                    idx_i, idx_j = np.triu_indices(n, k=1)
                    dx = centers_opt[:, 0][:, None] - centers_opt[:, 0][None, :]
                    dy = centers_opt[:, 1][:, None] - centers_opt[:, 1][None, :]
                    d2 = dx[idx_i, idx_j] ** 2 + dy[idx_i, idx_j] ** 2
                    rs2 = (radii_opt[idx_i] + radii_opt[idx_j]) ** 2
                    if np.any(d2 < rs2 - 1e-9):
                        valid = False
                        
                if valid:
                    s = np.sum(radii_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers_opt.copy()
                        best_radii = radii_opt.copy()
        except Exception:
            pass

    # Broad search
    for cfg in inits:
        try_opt(cfg)
        
    # Local refinement via perturbations
    if best_centers is not None:
        for _ in range(15):
            pert = np.clip(best_centers + rng.uniform(-0.005, 0.005, best_centers.shape), 0.05, 0.95)
            try_opt(pert)
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = make_hex([5, 6, 5, 6, 4])
        best_radii = solve_lp_radii(best_centers)
        if best_radii is None:
            best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict validity within numerical tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
