# sol_000150 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=90eb2ac0 sum of radii=2.609386 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_IDX = np.triu_indices(N, k=1)

def objective_t(vars_arr):
    """Objective for Phase 1: minimize negative equal radius t"""
    return -vars_arr[-1]

def constraints_t(vars_arr):
    """Constraints for Phase 1: boundary and pairwise separation for equal radius t"""
    t = vars_arr[-1]
    centers = vars_arr[:-1].reshape(N, 2)
    cons = []
    cons.append(centers[:, 0] - t)
    cons.append(1.0 - centers[:, 0] - t)
    cons.append(centers[:, 1] - t)
    cons.append(1.0 - centers[:, 1] - t)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    cons.append(dists[PAIR_IDX] - 2.0 * t)
    return np.concatenate(cons)

def solve_lp_radii(centers):
    """Phase 2: LP to maximize sum of radii for fixed centers"""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds = [(0.0, None)] * n
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(lim)
        
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def objective_joint(vars_arr):
    """Phase 3: minimize negative sum of radii"""
    return -np.sum(vars_arr[2 * N:])

def constraints_joint(vars_arr):
    """Phase 3: boundary and non-overlap constraints for variable radii"""
    cx = vars_arr[:N]
    cy = vars_arr[N:2 * N]
    r = vars_arr[2 * N:]
    
    c1 = cx - r
    c2 = 1.0 - cx - r
    c3 = cy - r
    c4 = 1.0 - cy - r
    
    cx_m = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_m = cy[:, np.newaxis] - cy[np.newaxis, :]
    r_m = r[:, np.newaxis] + r[np.newaxis, :]
    
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    
    c5 = d2[PAIR_IDX] - rs2[PAIR_IDX]
    
    return np.concatenate([c1, c2, c3, c4, c5])

def generate_initial_configs(rng):
    """Generate diverse hexagonal and grid starting positions"""
    configs = []
    
    # Hexagonal lattices with varying densities and rotations
    for r0 in np.linspace(0.085, 0.105, 6):
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 10:
            shift = r0 if row % 2 == 1 else 0.0
            x = r0 + shift
            while x + r0 <= 1.0:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        configs.append(pts)
        
        for rot in [0.1, -0.1, 0.25, -0.25, 0.4]:
            cos_t, sin_t = np.cos(rot), np.sin(rot)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rot_pts = pts @ R.T
            rot_pts -= rot_pts.min(axis=0)
            rot_pts /= rot_pts.max(axis=0)
            rot_pts = rot_pts * 0.88 + 0.06
            configs.append(rot_pts)
            
    # Regular grid + center
    grid = np.array([(i * 0.2 + 0.1, j * 0.2 + 0.1) for j in range(5) for i in range(5)])
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid)
    
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_t = [(0.0, 1.0)] * (2 * N) + [(0.05, 0.15)]
    cons_t = {'type': 'ineq', 'fun': constraints_t}
    
    best_centers = None
    best_sum = 0.0
    best_radii = None
    
    configs = generate_initial_configs(rng)
    
    # Phase 1: Optimize centers for maximum equal radius
    for cfg in configs:
        cfg_pert = cfg + rng.uniform(-0.008, 0.008, cfg.shape)
        cfg_pert = np.clip(cfg_pert, 0.02, 0.98)
        x0 = np.concatenate([cfg_pert.flatten(), [0.09]])
        
        try:
            res = minimize(objective_t, x0, method='SLSQP', bounds=bounds_t,
                           constraints=cons_t, options={'maxiter': 8000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_opt = res.x[:-1].reshape(N, 2)
                t_opt = res.x[-1]
                
                # Phase 2: LP for variable radii on optimized centers
                radii, s = solve_lp_radii(c_opt)
                if radii is not None and s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = radii.copy()
        except Exception:
            continue
            
    # Phase 3: Joint refinement of centers and radii
    if best_centers is not None:
        x0_joint = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii])
        bounds_joint = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
        cons_joint = {'type': 'ineq', 'fun': constraints_joint}
        
        try:
            res_j = minimize(objective_joint, x0_joint, method='SLSQP', bounds=bounds_joint,
                             constraints=cons_joint, options={'maxiter': 5000, 'ftol': 1e-13})
            if np.isfinite(res_j.fun):
                new_sum = -res_j.fun
                if new_sum > best_sum:
                    best_centers = np.column_stack((res_j.x[:N], res_j.x[N:2 * N]))
                    best_radii = res_j.x[2 * N:]
                    best_sum = new_sum
        except Exception:
            pass
            
    # Fallback if optimization fails
    if best_centers is None:
        r_fb = 0.095
        pts = []
        y = r_fb
        row = 0
        while len(pts) < N:
            shift = r_fb if row % 2 else 0
            x = r_fb + shift
            while x + r_fb <= 1.0 and len(pts) < N:
                pts.append([x, y])
                x += 2 * r_fb
            y += np.sqrt(3) * r_fb
            row += 1
        best_centers = np.array(pts[:N])
        best_radii = np.full(N, r_fb)
        best_sum = np.sum(best_radii)
        
    # Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
