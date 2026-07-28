# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000032 (state ac51bd1a) state=8071229d sum of radii=2.582479 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_constraints(x, n):
    """Computes all inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:3*n]
    
    num_con = 4 * n + n * (n - 1) // 2
    con = np.empty(num_con)
    idx = 0
    
    # Boundary constraints: circles inside [0,1]x[0,1]
    con[idx:idx+n] = cx - r
    idx += n
    con[idx:idx+n] = 1.0 - cx - r
    idx += n
    con[idx:idx+n] = cy - r
    idx += n
    con[idx:idx+n] = 1.0 - cy - r
    idx += n
    
    # Overlap constraints: squared distance >= squared sum of radii
    cx_col = cx[:, None]
    cy_col = cy[:, None]
    r_col = r[:, None]
    
    dist_sq = (cx_col - cx_col.T)**2 + (cy_col - cy_col.T)**2
    r_sum_sq = (r_col + r_col.T)**2
    
    triu_idx = np.triu_indices(n, k=1)
    con[idx:] = dist_sq[triu_idx] - r_sum_sq[triu_idx]
    
    return con

def objective_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:3*n])

def generate_initial_configs(n, rng):
    """Generates diverse initial center configurations."""
    configs = []
    
    # 1. Hexagonal lattices with different densities
    for r_guess in [0.07, 0.08, 0.09]:
        pts = []
        y = r_guess
        row = 0
        while len(pts) < n:
            shift = r_guess if row % 2 == 1 else 0.0
            x = r_guess + shift
            while x + r_guess <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r_guess
            y += r_guess * np.sqrt(3)
            row += 1
        pts = np.array(pts[:n])
        # Center and scale to fit comfortably inside [0,1]
        mx, my = pts.max(axis=0)
        mn, mny = pts.min(axis=0)
        scale = 0.88 / max(mx - mn, my - mny)
        pts = (pts - (mx + mn) / 2.0) * scale + 0.5
        configs.append(pts)
        
    # 2. Regular grid + center circle
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    grid_pts.append([0.5, 0.5])
    configs.append(np.array(grid_pts[:n]))
    
    # 3. Perturbed versions of the above to break symmetry
    for base in configs[:3]:
        for _ in range(3):
            cfg = base + rng.uniform(-0.025, 0.025, base.shape)
            cfg = np.clip(cfg, 0.05, 0.95)
            configs.append(cfg)
            
    return configs

def optimize_once(n, centers_init, bounds, cons):
    """Runs a single SLSQP optimization."""
    x0 = np.concatenate([centers_init.flatten(), np.full(n, 0.05)])
    try:
        res = minimize(
            objective_func, x0, args=(n,),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 3000, 'ftol': 1e-12}
        )
        return res.x
    except Exception:
        return None

def check_validity(centers, radii, n):
    """Fast validity check with strict tolerance."""
    cx, cy = centers.T
    r = radii
    if np.any(r < 0): return False
    if np.any(cx - r < -1e-9) or np.any(cx + r > 1.0 + 1e-9): return False
    if np.any(cy - r < -1e-9) or np.any(cy + r > 1.0 + 1e-9): return False
    
    dists = np.sqrt((cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2)
    np.fill_diagonal(dists, np.inf)
    r_sum = r[:, None] + r[None, :]
    if np.any(dists < r_sum - 1e-9):
        return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_res = None
    
    # Phase 1: Multi-start optimization
    inits = generate_initial_configs(n, rng)
    for cfg in inits:
        res_x = optimize_once(n, cfg, bounds, cons)
        if res_x is None:
            continue
            
        cx = res_x[:n]
        cy = res_x[n:2*n]
        r = res_x[2*n:3*n]
        centers = np.column_stack((cx, cy))
        
        if check_validity(centers, r, n):
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_res = res_x
                
    # Phase 2: Local perturbation search on the best found solution
    if best_res is not None:
        best_centers_init = best_res[:2*n].reshape(n, 2)
        for _ in range(8):
            perturbed = best_centers_init + rng.uniform(-0.015, 0.015, best_centers_init.shape)
            perturbed = np.clip(perturbed, 0.05, 0.95)
            
            res_x = optimize_once(n, perturbed, bounds, cons)
            if res_x is None:
                continue
                
            cx = res_x[:n]
            cy = res_x[n:2*n]
            r = res_x[2*n:3*n]
            centers = np.column_stack((cx, cy))
            
            if check_validity(centers, r, n):
                s = np.sum(r)
                if s > best_sum:
                    best_sum = s
                    best_res = res_x

    # Fallback (should rarely trigger)
    if best_res is None:
        fallback = np.zeros(3 * n)
        fallback[:n] = 0.5
        fallback[n:2*n] = 0.5
        fallback[2*n:] = 0.05
        best_res = fallback
        
    # Extract and prepare final solution
    cx_best = best_res[:n]
    cy_best = best_res[n:2*n]
    r_best = best_res[2*n:3*n]
    centers = np.column_stack((cx_best, cy_best))
    radii = r_best.copy()
    
    # Safety scaling to guarantee strict validity within grader tolerance
    scale = 1.0
    for i in range(n):
        if radii[i] < 1e-12: continue
        scale = min(scale, cx_best[i]/radii[i], (1.0-cx_best[i])/radii[i],
                    cy_best[i]/radii[i], (1.0-cy_best[i])/radii[i])
                    
    for i in range(n):
        for j in range(i + 1, n):
            if radii[i] + radii[j] < 1e-12: continue
            d = np.linalg.norm(centers[i] - centers[j])
            scale = min(scale, d / (radii[i] + radii[j]))
            
    # Apply shrinkage with minimal margin for numerical stability
    radii *= scale * 0.999998
    final_sum = np.sum(radii)
    
    return centers, radii, float(final_sum)
