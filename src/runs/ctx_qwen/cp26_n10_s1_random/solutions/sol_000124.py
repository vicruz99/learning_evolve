# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000066 (state 7dd8b726) state=b4c9fe10 sum of radii=2.434357 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Returns (radii_array, sum_radii) or (None, 0.0) on failure.
    """
    n = centers.shape[0]
    if n == 0:
        return None, 0.0
        
    # Distance to nearest boundary for each circle
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # LP: max sum(r) s.t. r_i + r_j <= dist_ij, 0 <= r_i <= wall_dists[i]
    # linprog minimizes c^T x, so we set c = -1
    c = -np.ones(n)
    
    # Constraints matrix A_ub x <= b_ub
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    bounds = [(0.0, max(0.0, w)) for w in wall_dists]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'disp': False})
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
        
    # Fallback safe assignment if LP fails (rare)
    r_fb = np.full(n, 1e-4)
    for i in range(n):
        min_d = wall_dists[i]
        for j in range(n):
            if i == j: continue
            min_d = min(min_d, dists[i, j])
        r_fb[i] = max(1e-4, min_d / 2.0)
    return r_fb, np.sum(r_fb)

def get_hex_configs(n, rng):
    """Generates multiple hexagonal lattice configurations with rotations and perturbations."""
    configs = []
    r_base = 0.09
    pts = []
    y = r_base
    row = 0
    while len(pts) < n + 10:
        shift = r_base if row % 2 == 1 else 0.0
        x = r_base + shift
        while x + r_base <= 1.0:
            pts.append([x, y])
            x += 2.0 * r_base
        y += np.sqrt(3) * r_base
        row += 1
    base_pts = np.array(pts[:n])
    configs.append(base_pts)
    
    # Rotated and scaled variants
    for angle in [0.15, 0.35, 0.6, 1.1]:
        pts_rot = base_pts.copy()
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        pts_rot = pts_rot @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts_rot -= pts_rot.min(axis=0)
        pts_rot /= pts_rot.max(axis=0)
        pts_rot = pts_rot * 0.88 + 0.12
        configs.append(pts_rot)
        
    # Perturbed variants
    for _ in range(4):
        pert = base_pts + rng.uniform(-0.025, 0.025, (n, 2))
        configs.append(np.clip(pert, 0.05, 0.95))
        
    return configs

def obj_para(vars_array, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_array[:n])

def con_para(vars_array, n, triu_idx):
    """Constraints: pairwise squared distance >= (r_i + r_j)^2"""
    r = vars_array[:n]
    u = vars_array[n:2*n]
    v = vars_array[2*n:3*n]
    
    # Parameterized coordinates guarantee boundary compliance
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    rs2 = rs**2
    
    return d2[triu_idx] - rs2[triu_idx]

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    rng = np.random.RandomState(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    configs = get_hex_configs(n, rng)
    
    # Phase 1: Parameterized Optimization to find good center skeletons
    for cfg in configs:
        r0 = np.full(n, 0.09)
        # Initialize u, v to map cfg coordinates back to [0,1]
        u0 = (cfg[:, 0] - r0) / (1.0 - 2.0 * r0)
        v0 = (cfg[:, 1] - r0) / (1.0 - 2.0 * r0)
        u0 = np.clip(u0, 0.0, 1.0)
        v0 = np.clip(v0, 0.0, 1.0)
        x0 = np.concatenate([r0, u0, v0])
        
        bounds = [(1e-4, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
        
        # Try base + 2 perturbations
        trials = [x0]
        for _ in range(2):
            xp = x0.copy()
            xp[n:] += rng.uniform(-0.02, 0.02, 2*n)
            xp[:n] *= rng.uniform(0.96, 1.04, n)
            xp = np.clip(xp, [1e-4]*n + [0.0]*(2*n), [0.5]*n + [1.0]*(2*n))
            trials.append(xp)
            
        for x0_try in trials:
            try:
                res = minimize(obj_para, x0_try, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': con_para, 'args': (n, triu_idx)},
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun):
                    r_opt = res.x[:n]
                    u_opt = res.x[n:2*n]
                    v_opt = res.x[2*n:3*n]
                    x_opt = r_opt + (1.0 - 2.0 * r_opt) * u_opt
                    y_opt = r_opt + (1.0 - 2.0 * r_opt) * v_opt
                    centers = np.column_stack((x_opt, y_opt))
                    
                    # Exact LP refinement for radii
                    lp_r, lp_s = solve_radii_lp(centers)
                    if lp_r is not None and lp_s > best_sum:
                        best_sum = lp_s
                        best_centers = centers.copy()
                        best_radii = lp_r.copy()
            except Exception:
                pass
                
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Phase 2: Coordinate-Ascent Local Search with LP evaluation
    centers = best_centers.copy()
    radii = best_radii.copy()
    current_sum = best_sum
    
    step_size = 0.015
    for iteration in range(2500):
        idx = rng.randint(0, n)
        new_centers = centers.copy()
        new_centers[idx] += rng.normal(0, step_size, 2)
        new_centers[idx] = np.clip(new_centers[idx], 0.0, 1.0)
        
        _, new_sum = solve_radii_lp(new_centers)
        if new_sum > current_sum:
            current_sum = new_sum
            centers = new_centers
            radii = solve_radii_lp(centers)[0]
            
        # Decay step size to fine-tune
        if iteration > 0 and iteration % 400 == 0:
            step_size *= 0.85
            
    # Phase 3: Strict Safety Validation & Scaling
    scale = 1.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 1e-12: continue
        scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            rs = radii[i] + radii[j]
            if rs < 1e-12: continue
            scale = min(scale, d/rs)
            
    # Apply scale with minimal margin for numerical safety
    radii *= max(scale * 0.999995, 0.0)
    
    # Final clipping to ensure absolute compliance
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    return centers, radii, float(np.sum(radii))
