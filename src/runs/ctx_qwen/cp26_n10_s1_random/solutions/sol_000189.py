# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000163 (state 5ceb6a50) state=3da80e81 sum of radii=2.622959 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective(vars_arr, n):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(vars_arr[2 * n:])

def get_constraints(vars_arr, n):
    """Computes inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2 * n]
    rs = vars_arr[2 * n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_boundary = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dr = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    idx = np.triu_indices(n, k=1)
    c_pairwise = (dx[idx]**2 + dy[idx]**2) - dr[idx]**2
    
    return np.concatenate([c_boundary, c_pairwise])

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

def generate_hex_config(n, row_counts, r_init):
    """Generates a hexagonal lattice configuration."""
    pts = []
    y = r_init
    for idx, cnt in enumerate(row_counts):
        shift = r_init if idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
    return np.array(pts[:n])

def generate_rotated_grid(n, angle_deg):
    """Generates a grid rotated by angle_deg degrees, then clipped/centered."""
    cols = int(np.ceil(np.sqrt(n)))
    pts = np.array([(c, r) for r in range(cols) for c in range(cols)]).astype(float)
    pts = pts[:n]
    
    # Normalize to unit square before rotation
    pts = pts / (cols - 1)
    
    # Rotate
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts = pts @ rot.T
    
    # Center and scale to fit roughly in [0.1, 0.9]
    pts -= pts.mean(axis=0)
    pts /= pts.max(axis=0).max() * 0.8
    pts += 0.5
    return pts

def force_pack(centers, n, steps=800):
    """Quick repulsive force simulation to tighten packing."""
    centers = centers.copy()
    r_target = 0.06
    dt = 0.002
    for _ in range(steps):
        forces = np.zeros_like(centers)
        # Wall forces
        for i in range(n):
            for d in range(2):
                if centers[i, d] - r_target < 0:
                    forces[i, d] += 10.0 * (r_target - centers[i, d])
                if centers[i, d] + r_target > 1.0:
                    forces[i, d] -= 10.0 * (centers[i, d] + r_target - 1.0)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-7
                overlap = 2.0 * r_target - dist
                if overlap > 0:
                    f = 5.0 * overlap / dist
                    forces[i, 0] += dx * f
                    forces[i, 1] += dy * f
                    forces[j, 0] -= dx * f
                    forces[j, 1] -= dy * f
        centers += forces * dt
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': get_constraints, 'args': (n,)}
    
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Hexagonal patterns
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [5, 6, 6, 5, 4], [5, 5, 5, 5, 6]
    ]
    for rd in row_dists:
        pts = generate_hex_config(n, rd, 0.08)
        inits.append(pts)
        inits.append(np.clip(pts + rng.uniform(-0.02, 0.02, pts.shape), 0.05, 0.95))
        
    # 2. Rotated grids
    for ang in [10, 15, 20, 25, 30]:
        inits.append(generate_rotated_grid(n, ang))
        
    # 3. Random + force packed
    for _ in range(5):
        c = rng.uniform(0.1, 0.9, (n, 2))
        inits.append(force_pack(c, n))
        
    # 4. Optimizer loop
    for cfg in inits:
        v0 = np.zeros(3 * n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = 0.06  # Feasible start
        
        try:
            res = minimize(objective, v0, args=(n,), method='SLSQP',
                           bounds=bounds_vars, constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                r = res.x[2*n:]
                centers = np.column_stack((cx, cy))
                
                # Phase 2: LP refinement on fixed centers
                r_lp, s_lp = solve_lp_radii(centers, n)
                if r_lp is not None and s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()
                    
        except Exception:
            continue
            
    # If we found a good configuration, refine it further
    if best_centers is not None:
        # Perturb and re-optimize jointly
        for _ in range(6):
            pert_c = np.clip(best_centers + rng.uniform(-0.005, 0.005, best_centers.shape), 0.05, 0.95)
            v0 = np.zeros(3 * n)
            v0[:n] = pert_c[:, 0]
            v0[n:2*n] = pert_c[:, 1]
            v0[2*n:] = best_radii * 0.95
            
            try:
                res = minimize(objective, v0, args=(n,), method='SLSQP',
                               bounds=bounds_vars, constraints=cons_dict,
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if np.isfinite(res.fun):
                    c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                    r_opt, s_opt = solve_lp_radii(c_opt, n)
                    if r_opt is not None and s_opt > best_sum:
                        best_centers = c_opt
                        best_radii = r_opt
                        best_sum = s_opt
            except Exception:
                pass
                
    # Fallback
    if best_centers is None:
        fallback = generate_hex_config(n, [5, 6, 5, 6, 4], 0.08)
        best_centers = fallback
        radii_fb, _ = solve_lp_radii(fallback, n)
        best_radii = radii_fb if radii_fb is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)

    # Strict safety scaling for checker tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
