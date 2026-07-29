# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000039 (state 91d6f1d3) state=1144c2b7 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Vectorized pairwise distance constraints
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((len(I_IDX), n))
    A_ub[np.arange(len(I_IDX)), I_IDX] = 1.0
    A_ub[np.arange(len(I_IDX)), J_IDX] = 1.0
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

def generate_init_centers(seed, style='hex'):
    """Generate structured or random initial center configurations."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    if style == 'hex':
        s = 0.14 + rng.uniform(0, 0.06)
        idx = 0
        row = 0
        y = s/2 + rng.uniform(0, 0.05)
        while idx < N and y < 1.0 - s/2:
            x_start = s/2 + (row % 2) * s/2 + rng.uniform(0, 0.02)
            col = 0
            while x_start + col*s < 1.0 - s/2 and idx < N:
                centers[idx] = [x_start + col*s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
    elif style == 'grid':
        step = 0.18 + rng.uniform(-0.02, 0.02)
        idx = 0
        y = step/2
        while y < 1.0 - step/2 and idx < N:
            x = step/2
            while x < 1.0 - step/2 and idx < N:
                centers[idx] = [x, y]
                x += step
                idx += 1
            y += step
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
    else:
        centers = rng.uniform(0.1, 0.9, (N, 2))
        
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Diverse initializations
    trials = []
    for seed in range(55):
        style = 'hex' if seed < 30 else 'grid' if seed < 42 else 'random'
        trials.append((generate_init_centers(seed, style), 0.0))
        
    # Add structured row patterns
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [6,6,6,6,2]]
    for pat in patterns:
        pts = []
        y = 0.05
        dy = 0.165
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.085
            x = 0.05 + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 0.17
            y += dy
        while len(pts) < N:
            pts.append([0.5, 0.5])
        trials.append((np.array(pts[:N]), 0.02))
        
    rng = np.random.RandomState(42)
    for trial_centers, noise_scale in trials:
        c0 = trial_centers + rng.normal(0, noise_scale, trial_centers.shape)
        c0 = np.clip(c0, 0.02, 0.98)
        
        r0, s0 = solve_lp_radii(c0)
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        # Joint optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                       options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
        
        curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
        curr_r, curr_s = solve_lp_radii(curr_c)
        
        if curr_s > best_sum:
            best_sum = curr_s
            best_centers = curr_c.copy()
            best_radii = curr_r.copy()
            
            # Local refinement
            for _ in range(3):
                cp = best_centers + rng.normal(0, 0.003, best_centers.shape)
                cp = np.clip(cp, 0.01, 0.99)
                rp, sp = solve_lp_radii(cp)
                xp = np.zeros(3*N)
                xp[0::3] = cp[:, 0]
                xp[1::3] = cp[:, 1]
                xp[2::3] = rp
                
                res2 = minimize(objective, xp, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                               options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                c2 = np.column_stack((res2.x[0::3], res2.x[1::3]))
                r2, s2 = solve_lp_radii(c2)
                if s2 > best_sum:
                    best_sum = s2
                    best_centers = c2.copy()
                    best_radii = r2.copy()
                    
    # Phase 2: Basin hopping / perturbation refinement
    if best_centers is not None:
        for _ in range(40):
            scale = rng.uniform(0.002, 0.015)
            cp = best_centers + rng.normal(0, scale, best_centers.shape)
            cp = np.clip(cp, 0.02, 0.98)
            rp, sp = solve_lp_radii(cp)
            if sp > 0:
                xp = np.zeros(3*N)
                xp[0::3] = cp[:, 0]
                xp[1::3] = cp[:, 1]
                xp[2::3] = rp
                res = minimize(objective, xp, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt, s_opt = solve_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
                    
    # Fallback safety net
    if best_centers is None:
        best_centers = generate_init_centers(0)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Strict post-processing to guarantee validity
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
