# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000154 (state 53d794eb) state=9c3936a2 sum of radii=2.626678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def constraints(x):
    """Returns all inequality constraints g(x) >= 0 (vectorized)."""
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

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
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
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.04)

def center_penalty(c_flat, radii):
    """Smooth penalty function for centers given fixed radii."""
    c = c_flat.reshape(-1, 2)
    p = 0.0
    n = c.shape[0]
    
    # Wall penalties
    x, y = c[:, 0], c[:, 1]
    r = radii
    p += np.sum(np.maximum(r - x, 0.0)**2)
    p += np.sum(np.maximum(x - (1.0 - r), 0.0)**2)
    p += np.sum(np.maximum(r - y, 0.0)**2)
    p += np.sum(np.maximum(y - (1.0 - r), 0.0)**2)
    
    # Overlap penalties (only upper triangle)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.hypot(dx, dy)
    gap = d - r[:, None] - r[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    violations = np.maximum(-gap[mask], 0.0)
    p += np.sum(violations**2)
    
    return p

def alternating_opt(centers, radii, n_iter=12):
    """Alternating LP (radii) and L-BFGS-B (centers) optimization."""
    c = centers.copy()
    r = radii.copy()
    bounds_c = [(0.001, 0.999)] * (2 * N)
    
    for _ in range(n_iter):
        r = solve_lp_radii(c)
        res = minimize(center_penalty, c.flatten(), args=(r,), 
                       method='L-BFGS-B', bounds=bounds_c, 
                       options={'maxiter': 800, 'ftol': 1e-14})
        c = res.x.reshape(-1, 2)
        c = np.clip(c, 0.005, 0.995)
        
    return c, r

def make_hex_init(seed, ang=0.0):
    """Generates a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    r0 = 0.09
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
    if ang != 0.0:
        center = np.array([0.5, 0.5])
        pts = (pts - center) @ np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) + center
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
    return pts[:N]

def make_force_init(seed):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for step in range(180):
        f = np.zeros_like(pts)
        lr = 0.04 * (1.0 - step / 180.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-5:
                    rep = 0.015 / (d**2 + 1e-4)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.06
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.06
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
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
    for s in range(12):
        inits.append(make_hex_init(s, ang=s * 0.04))
    for s in range(10):
        inits.append(make_force_init(s))
        
    # Phase 2: Multi-start Optimization (Alternating + SLSQP Polish)
    for pts in inits:
        c_curr, r_curr = pts.copy(), np.full(N, 0.065)
        
        # Alternating optimization to find a strong local topology
        c_curr, r_curr = alternating_opt(c_curr, r_curr, n_iter=12)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c_curr[:, 0]
        x0[1::3] = c_curr[:, 1]
        x0[2::3] = np.maximum(r_curr * 0.995, 1e-6)
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                # Quick LP snap on optimized centers to guarantee max radii
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
            continue

    # Phase 3: Aggressive Topology Search (Deflation & Repositioning)
    if best_x is not None:
        for cyc in range(70):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise = 0.0025 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 7)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.82
            
            x0 = project_to_feasible(x0)

            try:
                res = minimize(objective, x0, method='SLSQP', 
                               bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    centers_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(centers_p)
                    
                    x_p = res.x.copy()
                    x_p[2::3] = np.maximum(r_p * 0.998, 1e-7)
                    x_p = project_to_feasible(x_p)
                    
                    curr_sum = np.sum(x_p[2::3])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = x_p.copy()
            except Exception:
                pass

    # Phase 4: High-Precision Polish
    if best_x is not None:
        for _ in range(5):
            x_pol = best_x.copy()
            x_pol[2::3] *= 0.995
            x_pol = project_to_feasible(x_pol)
            try:
                res = minimize(objective, x_pol, method='SLSQP', bounds=bounds_obj, 
                               constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
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
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(np.sum(radii))
