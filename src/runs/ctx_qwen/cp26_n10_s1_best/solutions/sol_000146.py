# sol_000146 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000139 (state 7da59266) state=c35f696e sum of radii=1.560000 correctness=1.0
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
    """Returns all inequality constraints g(x) >= 0 using squared distances."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints
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
    n = N
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
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(r0, angle):
    """Generates a hexagonal lattice initialization with optional rotation."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 10:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
    return pts[:N]

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(200):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-5:
                    rep = 0.005 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.1: f[i, dim] += 0.05
                elif pts[i, dim] > 0.9: f[i, dim] -= 0.05
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def make_boundary_init():
    """Generates an initialization focused on corners and edges."""
    pts = []
    r = 0.1
    pts.extend([[r, r], [1-r, r], [r, 1-r], [1-r, 1-r]])
    for i in range(1, 4):
        pts.extend([[i*0.25, r], [i*0.25, 1-r], [r, i*0.25], [1-r, i*0.25]])
    pts.append([0.5, 0.5])
    while len(pts) < N:
        pts.append(np.random.uniform(0.2, 0.8, 2))
    return np.array(pts[:N])

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-7, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    for r0 in [0.085, 0.09, 0.095, 0.10, 0.105]:
        for ang in np.linspace(-0.35, 0.35, 9):
            inits.append(make_hex_init(r0, ang))
    for s in range(12):
        inits.append(make_force_init(s))
    for _ in range(6):
        inits.append(make_boundary_init())

    # Phase 2: Multi-start Optimization with LP-SLSQP Alternation
    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = r_lp * 0.99
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                if r_opt is not None:
                    x0[2::3] = r_opt
                    x0 = project_to_feasible(x0)
                    s_val = np.sum(x0[2::3])
                    if s_val > best_sum and np.min(constraints(x0)) >= -1e-5:
                        best_sum = s_val
                        best_x = x0.copy()
        except Exception:
            pass

    # Phase 3: Aggressive Topology Search & Refinement
    if best_x is not None:
        # 3a: Adaptive Noise Deflation to escape local minima
        for cyc in range(60):
            noise = 0.0025 * (0.92 ** cyc)
            x_p = best_x + np.random.normal(0, noise, 3 * N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 9)
            subset = np.random.choice(N, size=k, replace=False)
            x_p[subset * 3 + 2] *= 0.85
            
            x_p = project_to_feasible(x_p)
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = solve_lp_radii(c_opt)
                    if r_opt is not None:
                        x_p[2::3] = r_opt
                        x_p = project_to_feasible(x_p)
                        s_val = np.sum(x_p[2::3])
                        if s_val > best_sum and np.min(constraints(x_p)) >= -1e-5:
                            best_sum = s_val
                            best_x = x_p.copy()
            except Exception:
                pass

        # 3b: Global Rotation Search from best
        best_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-10, 10, 7):
            ang = np.deg2rad(ang_deg)
            c, s = np.cos(ang), np.sin(ang)
            rot_centers = (best_centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            rot_centers = np.clip(rot_centers, 0.01, 0.99)
            
            r_rot = solve_lp_radii(rot_centers)
            if r_rot is not None:
                x_rot = np.zeros(3 * N)
                x_rot[0::3] = rot_centers[:, 0]
                x_rot[1::3] = rot_centers[:, 1]
                x_rot[2::3] = r_rot
                x_rot = project_to_feasible(x_rot)
                
                try:
                    res = minimize(objective, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                                   options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun):
                        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_opt = solve_lp_radii(c_opt)
                        if r_opt is not None:
                            x_rot[2::3] = r_opt
                            x_rot = project_to_feasible(x_rot)
                            s_val = np.sum(x_rot[2::3])
                            if s_val > best_sum and np.min(constraints(x_rot)) >= -1e-5:
                                best_sum = s_val
                                best_x = x_rot.copy()
                except Exception:
                    pass

    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[2::3] = 0.06
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_sum = np.sum(best_x[2::3])

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()

    # Final strict validity adjustment against 1e-12 tolerance
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
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(np.sum(radii))
