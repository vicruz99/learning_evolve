# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000086 (state 1144c2b7) state=6e5e9bac sum of radii=2.619022 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(x):
    """Objective: maximize sum of radii (minimize negative sum)."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints g(x) >= 0:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Pairwise non-overlap
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    # Boundary clearance
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return res.x, -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0

def generate_inits(num_inits, seed):
    """Generate diverse initial center configurations."""
    rng = np.random.RandomState(seed)
    inits = []
    
    for i in range(num_inits):
        if i < 20:
            # Rotated hexagonal lattices
            s = 0.14 + rng.uniform(0, 0.08)
            angle = rng.uniform(-0.15, 0.15)
            pts = []
            row = 0
            y = s / 2
            while len(pts) < N:
                x = s / 2 + (row % 2) * s / 2
                while x < 1.0 - s / 2 and len(pts) < N:
                    # Rotate around center (0.5, 0.5)
                    xr = (x - 0.5) * np.cos(angle) - (y - 0.5) * np.sin(angle) + 0.5
                    yr = (x - 0.5) * np.sin(angle) + (y - 0.5) * np.cos(angle) + 0.5
                    pts.append([xr, yr])
                    x += s
                y += s * np.sqrt(3) / 2
                row += 1
            c = np.array(pts[:N]) + rng.normal(0, 0.004, (N, 2))
        elif i < 35:
            # Adaptive grid patterns
            step = 0.17 + rng.uniform(-0.02, 0.03)
            c = np.zeros((N, 2))
            idx = 0
            y = step / 2
            while y < 1.0 - step / 2 and idx < N:
                x = step / 2
                while x < 1.0 - step / 2 and idx < N:
                    c[idx] = [x, y]
                    x += step
                    idx += 1
                y += step
            while idx < N:
                c[idx] = rng.uniform(0.15, 0.85, 2)
                idx += 1
        else:
            # Random uniform
            c = rng.uniform(0.05, 0.95, (N, 2))
            
        inits.append(np.clip(c, 0.01, 0.99))
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Broad search with diverse initializations
    inits = generate_inits(65, seed=42)
    rng = np.random.RandomState(123)
    
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        r0 = np.maximum(r0 * 0.94, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 18000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                curr_r, curr_s = solve_lp_radii(curr_c)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            continue
            
    # Phase 2: Adaptive Basin Hopping & Polishing
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(100):
            # Decay noise schedule
            scale = 0.012 * np.exp(-step / 40.0)
            c_pert = current_c + rng.normal(0, scale, current_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > current_s:
                current_c, current_r, current_s = c_pert, r_pert, s_pert
                
                # Polish successful jump with SLSQP
                x0 = np.zeros(3 * N)
                x0[0::3] = current_c[:, 0]
                x0[1::3] = current_c[:, 1]
                x0[2::3] = np.maximum(current_r * 0.98, 1e-5)
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_opt, s_opt = solve_lp_radii(c_opt)
                        if s_opt > current_s:
                            current_c, current_r, current_s = c_opt, r_opt, s_opt
                            best_sum = current_s
                            best_centers = current_c.copy()
                            best_radii = current_r.copy()
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Strict post-processing to guarantee validator compliance
    radii = best_radii.copy()
    centers = best_centers.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(80):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc / 2.0
                    radii[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
