# sol_000241 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000177 (state 0ce77dda) state=5de10eaf sum of radii=2.628037 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def build_lp_matrix(n):
    """Builds the constant inequality matrix for the LP radius solver."""
    m = 4 * n + n * (n - 1) // 2
    A = np.zeros((m, n))
    k = 0
    for i in range(n):
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
        A[k, i] = 1.0; k += 1
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            k += 1
    return A

def solve_radii_lp(centers, A_ub):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    m = A_ub.shape[0]
    b = np.zeros(m)
    k = 0
    for i in range(n):
        x, y = centers[i]
        b[k] = x; k += 1
        b[k] = 1.0 - x; k += 1
        b[k] = y; k += 1
        b[k] = 1.0 - y; k += 1
    for i in range(n):
        for j in range(i + 1, n):
            b[k] = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            k += 1
            
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b, bounds=(0, None), method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-6)

def joint_objective(vars_flat, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_flat[2 * n:])

def joint_constraints(vars_flat, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = vars_flat[:n]
    cy = vars_flat[n:2 * n]
    r = vars_flat[2 * n:]
    
    c_list = []
    c_list.append(cx - r)
    c_list.append(1.0 - cx - r)
    c_list.append(cy - r)
    c_list.append(1.0 - cy - r)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = cx[i_idx] - cx[j_idx]
    dy = cy[i_idx] - cy[j_idx]
    dist = np.hypot(dx, dy)
    r_sum = r[i_idx] + r[j_idx]
    c_list.append(dist - r_sum)
    
    return np.concatenate(c_list)

def generate_hex_init(rows, r0, rng):
    """Generates initial positions on a hexagonal lattice with specified row distribution."""
    n = 26
    pts = []
    y = r0
    for ri, cnt in enumerate(rows):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < n:
        pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
    return np.array(pts[:n])

def optimize_stage(cfg, n, A_ub, bounds_vars):
    """Runs LP + SLSQP refinement pipeline and returns (centers, radii, sum_r)."""
    r_lp = solve_radii_lp(cfg, A_ub)
    s_lp = np.sum(r_lp)
    
    x0 = np.zeros(3 * n)
    x0[:n] = cfg[:, 0]
    x0[n:2 * n] = cfg[:, 1]
    x0[2 * n:] = r_lp
    
    try:
        res = minimize(joint_objective, x0, args=(n,), method='SLSQP',
                       bounds=bounds_vars,
                       constraints={'type': 'ineq', 'fun': joint_constraints, 'args': (n,)},
                       options={'maxiter': 8000, 'ftol': 1e-14})
        
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
                return centers_opt, radii_opt, np.sum(radii_opt)
                
    except Exception:
        pass
        
    return cfg, r_lp, s_lp

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    A_ub = build_lp_matrix(n)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [6, 6, 6, 4, 4], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6], [5, 6, 6, 5, 4]
    ]
    
    inits = []
    for p in patterns:
        if sum(p) >= n:
            inits.append(generate_hex_init(p, 0.09, rng))
            inits.append(np.clip(generate_hex_init(p, 0.09, rng) + rng.uniform(-0.02, 0.02, (n, 2)), 0.05, 0.95))
            
    for _ in range(15):
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Phase 1: Broad search from structured and random starts
    for cfg in inits:
        c_res, r_res, s_res = optimize_stage(cfg, n, A_ub, bounds_vars)
        if s_res > best_sum:
            best_sum = s_res
            best_centers = c_res.copy()
            best_radii = r_res.copy()
            
    # Phase 2: Local refinement via iterative perturbations
    if best_centers is not None:
        for _ in range(25):
            pert = np.clip(best_centers + rng.uniform(-0.005, 0.005, best_centers.shape), 0.05, 0.95)
            c_res, r_res, s_res = optimize_stage(pert, n, A_ub, bounds_vars)
            if s_res > best_sum:
                best_sum = s_res
                best_centers = c_res.copy()
                best_radii = r_res.copy()
                
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = generate_hex_init([5, 6, 5, 6, 4], 0.09, rng)
        best_radii = solve_radii_lp(best_centers, A_ub)
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
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
