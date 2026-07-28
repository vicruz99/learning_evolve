# sol_000191 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000163 (state 5ceb6a50) state=a5b3b1a4 sum of radii=2.631091 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective(vars_arr, n):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(vars_arr[2 * n:])

def constraints(vars_arr, n):
    """
    Computes inequality constraints >= 0 for valid packing.
    Layout: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    """
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for efficiency
    cx_m = cx[:, None]
    cy_m = cy[:, None]
    r_m = r[:, None]
    
    dx = cx_m - cx_m.T
    dy = cy_m - cy_m.T
    dr = r_m + r_m.T
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dist_sq = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2
    r_sum_sq = dr[i_idx, j_idx]**2
    
    c = np.concatenate([c, dist_sq - r_sum_sq])
    return c

def solve_lp_radii(centers, n):
    """Solves the LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 0.0)
    bounds = [(0.0, lim) for lim in lims]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx = np.triu_indices(n, k=1)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = dists[idx]
    
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def make_hex_init(n, rows_pattern, r0=0.09, rng=None):
    """Generates initial positions on a hexagonal lattice with specified row distribution."""
    pts = []
    y = r0
    for ri, cnt in enumerate(rows_pattern):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
            pts.append([x, y])
            x += 2 * r0
        y += np.sqrt(3) * r0
    while len(pts) < n:
        if rng is not None:
            pts.append(rng.uniform(r0, 1-r0, 2).tolist())
        else:
            pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Systematic hexagonal patterns
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [5, 5, 5, 6, 5], [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 4, 6],
        [5, 6, 5, 5, 5], [4, 5, 6, 5, 6], [5, 5, 5, 5, 4, 2]
    ]
    for p in patterns:
        if sum(p) < n: continue
        inits.append(make_hex_init(n, p, 0.082, rng))
        inits.append(make_hex_init(n, p, 0.092, rng))
        inits.append(make_hex_init(n, p, 0.102, rng))
        
    # 2. Grid + center variations
    g = np.linspace(0.14, 0.86, 5)
    grid = np.array([[x, y] for y in g for x in g])
    for extra in [[0.5, 0.5], [0.08, 0.08], [0.92, 0.08], [0.5, 0.08]]:
        inits.append(np.vstack([grid, extra]))
        
    # 3. Random dense starts
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    def run_opt(cfg, max_iter=5000):
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.055  # Start with feasible small radii
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_vars,
                           constraints=cons_dict, options={'maxiter': max_iter, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                cx, cy, r = res.x[:n], res.x[n:2*n], res.x[2*n:]
                centers = np.column_stack((cx, cy))
                
                # Strict validity check
                valid = True
                if np.any(centers[:, 0] < r - 1e-10) or np.any(centers[:, 0] > 1 - r + 1e-10) or \
                   np.any(centers[:, 1] < r - 1e-10) or np.any(centers[:, 1] > 1 - r + 1e-10):
                    valid = False
                if valid:
                    i_idx, j_idx = np.triu_indices(n, 1)
                    dx = centers[i_idx, 0] - centers[j_idx, 0]
                    dy = centers[i_idx, 1] - centers[j_idx, 1]
                    d2 = dx**2 + dy**2
                    rs = r[i_idx] + r[j_idx]
                    if np.any(d2 < rs**2 - 1e-9):
                        valid = False
                        
                if valid:
                    return centers.copy(), r.copy(), np.sum(r)
        except Exception:
            pass
        return None, None, 0.0

    # Phase 1: Broad search from diverse inits
    for cfg in inits:
        c, r, s = run_opt(cfg, max_iter=4000)
        if c is not None and s > best_sum:
            best_sum = s
            best_centers = c
            best_radii = r
            
            # Immediate LP refinement
            radii_lp, s_lp = solve_lp_radii(best_centers, n)
            if radii_lp is not None and s_lp > best_sum:
                best_radii = radii_lp
                best_sum = s_lp
                
    # Phase 2: Iterative relaxation & re-optimization to escape local minima
    if best_centers is not None:
        for _ in range(12):
            # Shrink radii to create maneuvering space
            shrunk_r = best_radii * 0.82
            # Perturb centers moderately
            pert_c = np.clip(best_centers + rng.uniform(-0.015, 0.015, (n, 2)), 0.03, 0.97)
            
            x0 = np.zeros(3*n)
            x0[:n] = pert_c[:, 0]
            x0[n:2*n] = pert_c[:, 1]
            x0[2*n:] = shrunk_r
            
            try:
                res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_vars,
                               constraints=cons_dict, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if np.isfinite(res.fun):
                    cx, cy, r = res.x[:n], res.x[n:2*n], res.x[2*n:]
                    centers = np.column_stack((cx, cy))
                    
                    # Validate
                    valid = True
                    if np.any(centers[:, 0] < r - 1e-10) or np.any(centers[:, 0] > 1 - r + 1e-10) or \
                       np.any(centers[:, 1] < r - 1e-10) or np.any(centers[:, 1] > 1 - r + 1e-10):
                        valid = False
                    if valid:
                        i_idx, j_idx = np.triu_indices(n, 1)
                        dx = centers[i_idx, 0] - centers[j_idx, 0]
                        dy = centers[i_idx, 1] - centers[j_idx, 1]
                        d2 = dx**2 + dy**2
                        rs = r[i_idx] + r[j_idx]
                        if np.any(d2 < rs**2 - 1e-9):
                            valid = False
                            
                    if valid:
                        s = np.sum(r)
                        # LP refine for fixed centers
                        radii_lp, s_lp = solve_lp_radii(centers, n)
                        if radii_lp is not None:
                            r_final = radii_lp
                            s_final = s_lp
                        else:
                            r_final = r
                            s_final = s
                            
                        if s_final > best_sum:
                            best_sum = s_final
                            best_centers = centers
                            best_radii = r_final
            except Exception:
                continue
                
    # Fallback configuration if optimization fails unexpectedly
    if best_centers is None:
        fallback = make_hex_init(n, [5, 6, 5, 6, 4], 0.085, rng)
        best_centers = fallback
        radii_fb, _ = solve_lp_radii(fallback, n)
        best_radii = radii_fb if radii_fb is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict numerical validity for the checker
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
