# sol_000156 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000141 (state 3805af16) state=d4922357 sum of radii=2.621364 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_hex_staggered_init(angle=0.0):
    """Generates a staggered hexagonal initialization (6-5-6-5-4 pattern)."""
    pts = []
    r0 = 0.092
    y = r0
    row_counts = [6, 5, 6, 5, 4]
    
    for idx, count in enumerate(row_counts):
        x = r0 + (6 - count) * r0 / 2.0  # Center the row
        for _ in range(count):
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        
    # Pad or trim
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    pts = np.array(pts[:N])
    
    if angle != 0.0:
        c = np.array([0.5, 0.5])
        pts = (pts - c) @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) + c
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]])
    return pts[:N]

def make_corner_edge_init():
    """Generates initialization focused on corners and edges."""
    pts = np.zeros((N, 2))
    r = 0.085
    # Corners
    pts[0] = [r, r]
    pts[1] = [1-r, r]
    pts[2] = [r, 1-r]
    pts[3] = [1-r, 1-r]
    # Edge mids
    pts[4] = [0.5, r]
    pts[5] = [0.5, 1-r]
    pts[6] = [r, 0.5]
    pts[7] = [1-r, 0.5]
    # Center
    pts[8] = [0.5, 0.5]
    
    idx = 9
    y = r + 0.06
    row = 0
    while idx < N and y + r <= 1.0:
        x = r + 0.06 if row % 2 == 0 else r + 0.18
        while idx < N and x + r <= 1.0:
            pts[idx] = [x, y]
            idx += 1
            x += 0.22
        y += 0.15
        row += 1
        
    for i in range(idx, N):
        pts[i] = np.random.uniform(0.15, 0.85, 2)
        
    pts += np.random.uniform(-0.005, 0.005, pts.shape)
    return np.clip(pts, 0.05, 0.95)

def force_layout_init(seed):
    """Generates initial configuration via repulsive force simulation."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    rs = np.full(N, 0.05)
    for step in range(300):
        f = np.zeros_like(pts)
        lr = 0.015 * (1.0 - step / 300.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.35 and d > 1e-5:
                    rep = 0.015 / (d**2 + 0.0005)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < rs[i] + 0.02: f[i, dim] += 0.05
                elif pts[i, dim] > 1.0 - rs[i] - 0.02: f[i, dim] -= 0.05
        pts += f * lr
        pts = np.clip(pts, 0.01, 0.99)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    
    # Staggered hex patterns with rotations
    for ang in np.linspace(-0.4, 0.4, 9):
        inits.append(make_hex_staggered_init(ang))
        
    # Corner/Edge focused
    for _ in range(6):
        inits.append(make_corner_edge_init())
        
    # Force-directed layouts
    for s in range(15):
        inits.append(force_layout_init(s))
        
    # Phase 2: Multi-start Optimization with LP-SLSQP Alternation
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        # Initialize radii via LP
        r_lp = solve_lp_radii(pts)
        x0[2::3] = np.maximum(r_lp * 0.98, 0.005)
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                if r_opt is not None:
                    curr_sum = np.sum(r_opt)
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_opt[:, 0]
                        best_x[1::3] = c_opt[:, 1]
                        best_x[2::3] = r_opt
        except Exception:
            pass
            
    # Phase 3: Aggressive Topology Search & Perturbation
    if best_x is not None:
        # 3a: Adaptive Noise Deflation
        for cyc in range(80):
            noise_scale = 0.003 * (0.90 ** cyc)
            x0 = best_x.copy()
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            # Deflate radii to create slack
            x0[2::3] *= 0.94
            
            # Aggressively deflate a random subset to break rigid contact networks
            subset = np.random.choice(N, size=max(4, N // 3), replace=False)
            x0[subset * 3 + 2] *= 0.70
            
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x_p = res.x.copy()
                        x_p[2::3] = r_new * 0.998
                        x_p = project_to_feasible(x_p)
                        
                        s_val = np.sum(x_p[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_x = x_p.copy()
            except Exception:
                pass
                
        # 3b: Global Rotation Search from best
        best_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-15, 15, 11):
            ang = np.deg2rad(ang_deg)
            c, s = np.cos(ang), np.sin(ang)
            rot_centers = (best_centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            rot_centers = np.clip(rot_centers, 0.01, 0.99)
            
            r_rot = solve_lp_radii(rot_centers)
            if r_rot is not None and np.sum(r_rot) > best_sum:
                x_rot = np.zeros(3 * N)
                x_rot[0::3] = rot_centers[:, 0]
                x_rot[1::3] = rot_centers[:, 1]
                x_rot[2::3] = r_rot
                x_rot = project_to_feasible(x_rot)
                
                try:
                    res = minimize(objective, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                                   options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun):
                        c_rot = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_final = solve_lp_radii(c_rot)
                        if r_final is not None:
                            x_rot[2::3] = r_final
                            x_rot = project_to_feasible(x_rot)
                            s_rot = np.sum(x_rot[2::3])
                            if s_rot > best_sum:
                                best_sum = s_rot
                                best_x = x_rot.copy()
                except Exception:
                    pass

    # Phase 4: High-Precision Polish
    if best_x is not None:
        for _ in range(10):
            x_pol = best_x.copy()
            x_pol[2::3] *= 0.995
            x_pol = project_to_feasible(x_pol)
            try:
                res = minimize(objective, x_pol, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_pol = solve_lp_radii(c_pol)
                    if r_pol is not None:
                        s_pol = np.sum(r_pol)
                        if s_pol > best_sum:
                            best_sum = s_pol
                            best_x = np.zeros(3 * N)
                            best_x[0::3] = c_pol[:, 0]
                            best_x[1::3] = c_pol[:, 1]
                            best_x[2::3] = r_pol
            except Exception:
                pass

    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        best_sum = np.sum(best_x[2::3])
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final strict validation repair against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or \
               centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
