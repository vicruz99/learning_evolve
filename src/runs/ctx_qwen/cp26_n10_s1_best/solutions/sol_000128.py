# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000119 (state 4956bde4) state=80db378b sum of radii=2.634292 correctness=1.0
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
    return [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N

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
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-6)
    except Exception:
        pass
    return np.full(n, 0.05)

def force_spread(centers, radii, steps=300):
    """Force-directed relaxation to spread points evenly and push to boundaries."""
    pts = centers.copy()
    rs = radii.copy()
    for step in range(steps):
        f = np.zeros_like(pts)
        lr = 0.015 * (1.0 - step / steps)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                target = rs[i] + rs[j] + 0.01
                if d < target and d > 1e-5:
                    rep = 0.02 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < rs[i] + 0.01: f[i, dim] += 0.08
                elif pts[i, dim] > 1.0 - rs[i] - 0.01: f[i, dim] -= 0.08
        pts += f * lr
        pts = np.clip(pts, 0.001, 0.999)
    return pts

def make_patterned_init(rows, angle=0.0):
    """Generates a row-patterned initialization with optional rotation."""
    pts = []
    r_est = 0.09
    y = r_est
    row_idx = 0
    for count in rows:
        x_start = r_est if row_idx % 2 == 0 else 2.0 * r_est
        for _ in range(count):
            if len(pts) >= N: break
            pts.append([x_start, y])
            x_start += 2.0 * r_est
        y += np.sqrt(3) * r_est
        row_idx += 1
        if len(pts) >= N: break
        
    # Pad if necessary
    while len(pts) < N:
        pts.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        
    pts = np.array(pts[:N])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
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
    
    # Patterned rows (breaks symmetry of pure hex grids)
    for rows in [[6,5,6,5,4], [5,6,6,5,4], [4,6,6,6,4], [6,6,5,5,4], [5,5,6,6,4], [6,7,6,7]]:
        for ang in np.linspace(-0.25, 0.25, 7):
            inits.append(make_patterned_init(rows, ang))
            
    # Hex grid variants
    for r0 in [0.085, 0.09, 0.095, 0.10]:
        for ang in np.linspace(-0.3, 0.3, 9):
            pts = make_patterned_init([5]*6 + [1], ang)
            inits.append(pts)
            
    # Force spread random layouts
    for s in range(10):
        np.random.seed(s + 200)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        inits.append(force_spread(pts, np.full(N, 0.05), steps=200))
        
    # Phase 2: Multi-start Optimization with LP Initialization
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        # Initialize radii via LP for maximum head-start
        r_lp = solve_lp_radii(pts)
        x0[2::3] = np.maximum(r_lp * 0.99, 0.005)
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                s = -res.fun
                vals = constraints(res.x)
                if np.min(vals) >= -1e-5 and s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cyc in range(60):
            x0 = best_x.copy()
            # Cooling perturbation
            noise = 0.002 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate radii to allow repositioning
            x0[2::3] *= 0.96
            
            # Aggressively deflate a random circle to break rigid contact networks
            idx = np.random.randint(N)
            x0[idx * 3 + 2] *= 0.4
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    # LP refinement on new centers to snap radii to theoretical max
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x_p = res.x.copy()
                        x_p[2::3] = r_new * 0.999
                        x_p = project_to_feasible(x_p)
                        
                        s = np.sum(x_p[2::3])
                        vals = constraints(x_p)
                        if np.min(vals) >= -1e-5 and s > best_sum:
                            best_sum = s
                            best_x = x_p.copy()
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
    radii = best_x[2::3]
    
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
            
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
