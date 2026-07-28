# sol_000212 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000165 (state ab534a56) state=7e0c373c sum of radii=2.626571 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_constraints(vars_arr, n):
    """Compute inequality constraints >= 0 for valid packing."""
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    b1 = cx - r
    b2 = 1.0 - cx - r
    b3 = cy - r
    b4 = 1.0 - cy - r
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    d2 = cx_diff**2 + cy_diff**2
    rs2 = r_sum**2
    
    idx = np.triu_indices(n, k=1)
    p_cons = d2[idx] - rs2[idx]
    
    return np.concatenate([b1, b2, b3, b4, p_cons])

def objective_func(vars_arr, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m_bound = 4 * n
    m_pair = n * (n - 1) // 2
    
    A_ub = np.zeros((m_bound + m_pair, n))
    b_ub = np.zeros(m_bound + m_pair)
    bounds = []
    
    idx = 0
    for i in range(n):
        x, y = centers[i]
        mx = max(1e-9, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            A_ub[idx, i] = 1.0
            b_ub[idx] = lim
            idx += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def generate_hex_starts(n, rng):
    """Generates hexagonal lattice initial configurations."""
    starts = []
    row_configs = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 5, 5, 6, 4],
        [5, 6, 6, 4, 5], [5, 5, 5, 5, 6], [7, 6, 6, 7]
    ]
    for rc in row_configs:
        if sum(rc) < n: 
            continue
        pts = []
        r0 = 0.09
        y = r0
        for i, cnt in enumerate(rc):
            shift = r0 if i % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        while len(pts) < n:
            pts.append([0.5, 0.5])
            
        pts = np.array(pts[:n])
        # Center configuration in the square
        cx_m, cy_m = pts.mean(axis=0)
        pts -= np.array([cx_m - 0.5, cy_m - 0.5])
        pts = np.clip(pts, 0.05, 0.95)
        starts.append(pts)
        
        # Add perturbations to break symmetry
        for _ in range(4):
            pert = pts + rng.uniform(-0.02, 0.02, pts.shape)
            starts.append(np.clip(pert, 0.05, 0.95))
    return starts

def force_directed_init(n, rng):
    """Generates initial configuration via repulsive force simulation."""
    centers = rng.uniform(0.2, 0.8, (n, 2))
    r = 0.06
    step = 0.05
    for _ in range(300):
        forces = np.zeros_like(centers)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r + 0.01 and dist > 1e-6:
                    f = (2 * r - dist) * d / dist
                    forces[i] += f
                    forces[j] -= f
        # Boundary repulsion
        for i in range(n):
            if centers[i, 0] < r: forces[i, 0] += (r - centers[i, 0])
            elif centers[i, 0] > 1 - r: forces[i, 0] -= (centers[i, 0] - (1 - r))
            if centers[i, 1] < r: forces[i, 1] += (r - centers[i, 1])
            elif centers[i, 1] > 1 - r: forces[i, 1] -= (centers[i, 1] - (1 - r))
            
        centers += step * forces
        centers = np.clip(centers, 0.02, 0.98)
        step *= 0.995
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(123)
    
    # Generate diverse initial configurations
    inits = generate_hex_starts(n, rng)
    for _ in range(6):
        inits.append(force_directed_init(n, rng))
    for _ in range(12):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_opt = [(0.001, 0.999)] * (2 * n) + [(0.0001, 0.25)] * n
    cons_dict = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multi-start SLSQP + LP Refinement
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        # Estimate safe initial radii
        dists_wall = np.minimum(np.minimum(cfg[:, 0], 1.0 - cfg[:, 0]), 
                                np.minimum(cfg[:, 1], 1.0 - cfg[:, 1]))
        min_wall = np.min(dists_wall)
        x0[2*n:] = min(0.1, min_wall) * 0.8
        
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP', 
                           bounds=bounds_opt, constraints=cons_dict,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            if not np.isfinite(res.fun):
                continue
                
            c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
            r_lp = solve_lp_radii(c_opt)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative local refinement around best configuration
    if best_centers is not None:
        for trial in range(25):
            # Decaying perturbation magnitude
            noise_scale = 0.004 * (0.95**trial)
            pert = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            
            x0 = np.zeros(3 * n)
            x0[:n] = pert[:, 0]
            x0[n:2*n] = pert[:, 1]
            x0[2*n:] = best_radii * 0.95
            
            try:
                res = minimize(objective_func, x0, args=(n,), method='SLSQP', 
                               bounds=bounds_opt, constraints=cons_dict,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                               
                if not np.isfinite(res.fun):
                    continue
                    
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
            except Exception:
                continue

    # Fallback configuration
    if best_centers is None:
        best_centers = inits[0]
        best_radii = np.full(n, 0.085)
        best_sum = np.sum(best_radii)
        
    # Final safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
