# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000118 (state 224f6ad6) state=d5d08215 sum of radii=2.635983 correctness=1.0
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
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
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
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A[idx, i] = 1.0
        b[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, d)
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def project(x):
    """Project variables to strictly satisfy bounds."""
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_hex_init(r0, ang):
    """Generates a hexagonal lattice initialization with optional rotation."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    pts = np.array(pts[:N])
    if ang != 0.0:
        c, s = np.cos(ang), np.sin(ang)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    return pts

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(150):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-5:
                    rep = 0.008 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.04
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.04
        pts += f * 0.04
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    best_sum = -1.0
    best_x = None

    # Phase 1: Diverse Initial Configurations
    inits = []
    for r0 in [0.085, 0.09, 0.095, 0.10]:
        for ang in np.linspace(-0.25, 0.25, 9):
            inits.append(make_hex_init(r0, ang))
    for s in range(12):
        inits.append(make_force_init(s))

    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = solve_lp_radii(pts) * 0.995
        x0 = project(x0)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                s_val = -res.fun
                if s_val > best_sum and np.min(constraints(res.x)) >= -1e-5:
                    best_sum = s_val
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Iterative Deflation & Refinement to Escape Local Minima
    if best_x is not None:
        for cycle in range(50):
            x0 = best_x.copy()
            # Gradually recover radii shrinkage to allow smoother topology changes
            shrink = 0.80 + 0.20 * (cycle / 50.0)
            x0[2::3] *= shrink
            noise = 0.003 / (cycle + 1)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            x0 = project(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    x0[2::3] = r_new * 0.998
                    x0 = project(x0)
                    s_val = np.sum(x0[2::3])
                    if s_val > best_sum and np.min(constraints(x0)) >= -1e-5:
                        best_sum = s_val
                        best_x = x0.copy()
            except Exception:
                pass

    # Fallback initialization
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final LP squeeze to maximize radii for the best centers found
    final_r = solve_lp_radii(centers)
    if np.sum(final_r) > np.sum(radii) - 1e-7:
        radii = final_r.copy()

    # Final strict validation repair
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
