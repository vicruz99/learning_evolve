# sol_000154 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000141 (state 3805af16) state=53d794eb sum of radii=2.629515 correctness=1.0
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
    """Returns all inequality constraints g(x) >= 0."""
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

def make_init_strategy(strategy, seed):
    """Generates diverse initial configurations."""
    np.random.seed(seed)
    if strategy == 'hex':
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
        
        ang = np.random.uniform(-0.4, 0.4)
        c = np.array([0.5, 0.5])
        pts = (pts - c) @ np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) + c
        
        mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
        pts = pts[mask]
        while len(pts) < N:
            pts = np.vstack([pts, [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]])
        return pts[:N]
        
    elif strategy == 'force':
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.3 and d > 1e-5:
                        rep = 0.01 / (d**2 + 1e-4)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
                for dim in range(2):
                    if pts[i, dim] < 0.15: f[i, dim] += 0.03
                    elif pts[i, dim] > 0.85: f[i, dim] -= 0.03
            pts += f * 0.04
            pts = np.clip(pts, 0.02, 0.98)
        return pts
        
    elif strategy == 'boundary':
        pts = []
        r = 0.085
        # Corners
        for cx in [r, 1.0 - r]:
            for cy in [r, 1.0 - r]:
                pts.append([cx, cy])
        # Edges
        for cx in np.linspace(r, 1.0 - r, 6)[1:-1]:
            pts.append([cx, r])
            pts.append([cx, 1.0 - r])
        for cy in np.linspace(r, 1.0 - r, 6)[1:-1]:
            pts.append([r, cy])
            pts.append([1.0 - r, cy])
        # Fill remaining with staggered pattern
        y = r + 0.05
        row = 0
        while len(pts) < N:
            x = r + 0.05 if row % 2 == 0 else 2.0 * r + 0.05
            while x <= 1.0 - r and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r
            y += np.sqrt(3.0) * r
            row += 1
        return np.array(pts[:N])
        
    return np.random.uniform(0.1, 0.9, (N, 2))

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multi-start with Diverse Initializations
    inits = []
    for s in range(12):
        inits.append(make_init_strategy('hex', s))
    for s in range(8):
        inits.append(make_init_strategy('force', s))
    for s in range(5):
        inits.append(make_init_strategy('boundary', s))
        
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        r_lp = solve_lp_radii(pts)
        x0[2::3] = r_lp * 0.99
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13})
            if not np.isnan(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                if r_opt is not None:
                    s_val = np.sum(r_opt)
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_opt[:, 0]
                        best_x[1::3] = c_opt[:, 1]
                        best_x[2::3] = r_opt
        except Exception:
            pass

    # Phase 2: Basin Hopping / Deflation to Escape Local Minima
    if best_x is not None:
        for step in range(80):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise = 0.0025 * (0.93 ** step)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(2, 7)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.80
            
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x_p = res.x.copy()
                        x_p[2::3] = r_new * 0.999
                        x_p = project_to_feasible(x_p)
                        
                        s_val = np.sum(x_p[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_x = x_p.copy()
            except Exception:
                pass

    # Phase 3: High-Precision Polish
    if best_x is not None:
        for _ in range(6):
            x_pol = best_x.copy()
            x_pol[2::3] *= 0.99
            x_pol = project_to_feasible(x_pol)
            try:
                res = minimize(objective, x_pol, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-14})
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
