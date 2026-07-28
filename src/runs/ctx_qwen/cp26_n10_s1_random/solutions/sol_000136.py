# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=2d560cd8 sum of radii=2.594189 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_min_clearance(centers_flat):
    """Computes half the minimum distance to any boundary or other circle center."""
    n = N_CIRCLES
    cx = centers_flat.reshape(n, 2)
    wall_dists = np.minimum(
        np.minimum(cx[:, 0], 1.0 - cx[:, 0]),
        np.minimum(cx[:, 1], 1.0 - cx[:, 1])
    )
    min_wall = np.min(wall_dists)
    
    diff = cx[:, np.newaxis, :] - cx[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    wall_dists = np.maximum(wall_dists, 0.0)
    bounds = [(0.0, w) for w in wall_dists]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return None, 0.0

def nelder_mead_obj(centers_flat):
    """Objective for Phase 1: maximize min clearance."""
    return -compute_min_clearance(centers_flat)

def slsqp_obj(v):
    """Objective for Phase 3: maximize sum of radii."""
    return -np.sum(v[2::3])

def slsqp_cons(v):
    """Inequality constraints for Phase 3: boundary and non-overlap."""
    n = N_CIRCLES
    cx, cy, r = v[0::3], v[1::3], v[2::3]
    c = []
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    d2 = dx**2 + dy**2
    r2 = dr**2
    idx = np.triu_indices(n, 1)
    c.extend(d2[idx] - r2[idx])
    return np.array(c)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    rng = np.random.default_rng(42)
    
    # Hexagonal row patterns summing to 26
    row_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4], [5, 5, 5, 5, 6]
    ]
    
    inits = []
    for pattern in row_patterns:
        pts = []
        y = 0.1
        r0 = 0.101
        for r_idx, count in enumerate(pattern):
            if len(pts) >= n: break
            shift = r0 if r_idx % 2 == 1 else 0.0
            row_span = (count - 1) * 2 * r0
            x_start = 0.5 - row_span / 2.0 + shift
            for c in range(count):
                if len(pts) >= n: break
                x = x_start + c * 2 * r0
                pts.append([x, y])
            y += np.sqrt(3) * r0
        pts = np.array(pts[:n])
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            pts = pts[idx]
        inits.append(np.clip(pts, 0.05, 0.95))
        
    # Add perturbed versions to escape symmetry
    for _ in range(5):
        p = inits[0].copy() + rng.uniform(-0.03, 0.03, inits[0].shape)
        inits.append(np.clip(p, 0.05, 0.95))
        
    best_centers = None
    best_sum = 0.0
    best_radii = None
    
    # Phase 1 & 2: Optimize centers then solve LP for radii
    for init_cfg in inits:
        res = minimize(nelder_mead_obj, x0=init_cfg.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        opt_centers = res.x.reshape(n, 2)
        opt_centers = np.clip(opt_centers, 0.0, 1.0)
        
        radii, s = solve_lp_radii(opt_centers)
        if radii is not None and s > best_sum:
            best_sum = s
            best_centers = opt_centers.copy()
            best_radii = radii.copy()
            
    # Phase 3: Joint refinement with SLSQP
    if best_centers is not None:
        x0 = np.zeros(3 * n)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_radii
        
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
        
        for _ in range(3):
            try:
                res2 = minimize(slsqp_obj, x0, method='SLSQP', bounds=bounds,
                                constraints={'type': 'ineq', 'fun': slsqp_cons},
                                options={'maxiter': 10000, 'ftol': 1e-12})
                if np.isfinite(res2.fun):
                    v_opt = res2.x
                    c_vals = slsqp_cons(v_opt)
                    if np.min(c_vals) > -1e-9:
                        s_opt = np.sum(v_opt[2::3])
                        if s_opt > best_sum:
                            best_sum = s_opt
                            best_centers = np.column_stack((v_opt[0::3], v_opt[1::3]))
                            best_radii = v_opt[2::3].copy()
                # Perturb slightly for next restart to explore nearby basins
                x0 += rng.normal(0, 0.001, x0.shape)
                x0 = np.clip(x0, 0.0, 1.0)
            except Exception:
                continue
                
    # Safety scaling to strictly satisfy the 1e-12 grader tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
