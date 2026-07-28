# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=13a30ca7 sum of radii=2.505333 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def equal_radius_objective(vars_arr):
    """Objective: minimize negative equal radius R"""
    return -vars_arr[-1]

def compute_equal_constraints(vars_arr, n):
    """
    Computes inequality constraints >= 0 for equal-radius packing.
    Variables: [x1, y1, ..., xn, yn, R]
    """
    centers = vars_arr[:2 * n].reshape(n, 2)
    R = vars_arr[2 * n]
    
    c = []
    # Boundary constraints: x >= R, 1-x >= R, y >= R, 1-y >= R
    c.append(centers[:, 0] - R)
    c.append(1.0 - centers[:, 0] - R)
    c.append(centers[:, 1] - R)
    c.append(1.0 - centers[:, 1] - R)
    
    # Pairwise non-overlap: dist(i,j) >= 2R
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, 1.0)  # Ignore self-distances
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.append(dist[mask] - 2.0 * R)
    
    return np.concatenate(c)

def solve_lp_radii(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Constraints: r_i + r_j <= dist(i,j) and boundary limits.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)  # Maximize sum(r) => Minimize -sum(r)
    bounds = [(0.0, None)] * n
    
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    m = len(pairs) + 4 * n
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i, j in pairs:
        d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(n):
        x, y = centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def generate_hex_pattern(row_counts, R):
    """Generates initial center positions on a hexagonal lattice."""
    n_target = 26
    pts = []
    y = R
    for ri, cnt in enumerate(row_counts):
        shift = R if ri % 2 == 1 else 0.0
        x = R + shift
        for _ in range(cnt):
            if len(pts) >= n_target:
                break
            pts.append([x, y])
            x += 2.0 * R
        y += np.sqrt(3) * R
    while len(pts) < n_target:
        pts.append([0.5, 0.5])
    return np.array(pts[:n_target])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Known good row distributions for N=26 in a square
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [4, 5, 6, 5, 6], [6, 4, 5, 6, 5]
    ]
    
    inits = []
    for p in patterns:
        if sum(p) < n:
            continue
        pts = generate_hex_pattern(p, 0.09)
        inits.append(pts)
        # Add perturbations to break symmetry
        for _ in range(3):
            pert = pts + rng.uniform(-0.02, 0.02, pts.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # Random starts
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_eq = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.12)]
    cons_eq = {'type': 'ineq', 'fun': compute_equal_constraints, 'args': (n,)}
    
    # Phase 1: Broad search over initial configurations
    for cfg in inits:
        x0 = np.concatenate([cfg.flatten(), [0.08]])
        try:
            res = minimize(equal_radius_objective, x0, method='SLSQP', bounds=bounds_eq,
                           constraints=cons_eq, options={'maxiter': 5000, 'ftol': 1e-12})
            
            # Accept if feasible or significantly better
            c_vals = compute_equal_constraints(res.x, n)
            if np.min(c_vals) >= -1e-6 or -res.fun > best_sum / n:
                c_opt = res.x[:2 * n].reshape(n, 2)
                r_lp, s_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    r_lp *= 0.9999999
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement around the best found configuration
    if best_centers is not None:
        for _ in range(15):
            # Perturb best centers slightly
            pert_c = np.clip(best_centers + rng.uniform(-0.008, 0.008, best_centers.shape), 0.05, 0.95)
            x0 = np.concatenate([pert_c.flatten(), [0.07]])
            try:
                res = minimize(equal_radius_objective, x0, method='SLSQP', bounds=bounds_eq,
                               constraints=cons_eq, options={'maxiter': 3000, 'ftol': 1e-12})
                c_ref = res.x[:2 * n].reshape(n, 2)
                r_lp, s_lp = solve_lp_radii(c_ref)
                if r_lp is not None:
                    r_lp *= 0.9999999
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_ref.copy()
                        best_radii = r_lp.copy()
            except Exception:
                continue
                
    # Fallback configuration if optimization yields nothing valid
    if best_centers is None:
        best_centers = generate_hex_pattern([5, 6, 5, 6, 4], 0.09)
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Final safety scaling to guarantee strict validity against 1e-12 tolerance
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
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
