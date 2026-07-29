# sol_000113 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000069 (state 13ab459c) state=71a210f9 sum of radii=2.627905 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_fun(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        xs - rs, 1.0 - xs - rs,
        ys - rs, 1.0 - ys - rs
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

def solve_lp(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
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
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_fun}
    best_sum = -1.0
    best_x = None
    np.random.seed(42)

    # Phase 1: Diverse Initializations
    inits = []
    
    # Rotated Hexagonal Lattices
    for seed in range(25):
        r0 = 0.085 + seed * 0.002
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
        ang = seed * 0.08
        ca, sa = np.cos(ang), np.sin(ang)
        pts = (pts - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
        pts += np.random.normal(0, 0.002, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)
        
    # Force-Directed Spreads
    for _ in range(15):
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(300):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[j] - pts[i]
                    dist = np.hypot(diff[0], diff[1])
                    if dist < 0.3 and dist > 1e-5:
                        force = 0.001 / (dist**2 + 1e-4)
                        f[i] -= diff * force / dist
                        f[j] += diff * force / dist
            pts += f * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 2: Multi-Start SLSQP with LP Warm-Start
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        r_lp = solve_lp(pts)
        x0[2::3] = np.maximum(r_lp * 0.99, 0.005)
        
        for i in range(N):
            r = x0[3 * i + 2]
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                vals = constraint_fun(res.x)
                if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Local Search (Subset Perturbation)
    if best_x is not None:
        for step in range(50):
            x_curr = best_x.copy()
            k = np.random.randint(1, 6)
            idxs = np.random.choice(N, k, replace=False)
            noise = 0.006 * (0.96 ** step)
            
            x_curr[3 * idxs] += np.random.normal(0, noise, k)
            x_curr[3 * idxs + 1] += np.random.normal(0, noise, k)
            x_curr[2::3] *= 0.980
            
            for i in range(N):
                r = max(0.005, x_curr[3 * i + 2])
                x_curr[3 * i] = np.clip(x_curr[3 * i], r, 1.0 - r)
                x_curr[3 * i + 1] = np.clip(x_curr[3 * i + 1], r, 1.0 - r)
                x_curr[3 * i + 2] = r
                
            try:
                res = minimize(objective, x_curr, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun):
                    vals = constraint_fun(res.x)
                    if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                        best_x = res.x.copy()
                        best_sum = -res.fun
                        c_tmp = np.column_stack((best_x[0::3], best_x[1::3]))
                        r_new = solve_lp(c_tmp)
                        best_x[2::3] = r_new
                        best_sum = np.sum(r_new)
            except Exception:
                pass

    # Phase 4: Global Rotation Search (Symmetry Breaking)
    if best_x is not None:
        for _ in range(20):
            x_try = best_x.copy()
            ang = np.random.uniform(-0.2, 0.2)
            ca, sa = np.cos(ang), np.sin(ang)
            
            cx = x_try[0::3] - 0.5
            cy = x_try[1::3] - 0.5
            x_try[0::3] = cx * ca - cy * sa + 0.5
            x_try[1::3] = cx * sa + cy * ca + 0.5
            x_try[2::3] *= 0.97
            
            for i in range(N):
                r = max(0.005, x_try[3 * i + 2])
                x_try[3 * i] = np.clip(x_try[3 * i], r, 1.0 - r)
                x_try[3 * i + 1] = np.clip(x_try[3 * i + 1], r, 1.0 - r)
                x_try[3 * i + 2] = r
                
            try:
                res = minimize(objective, x_try, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun):
                    vals = constraint_fun(res.x)
                    if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                        best_x = res.x.copy()
                        best_sum = -res.fun
                        c_tmp = np.column_stack((best_x[0::3], best_x[1::3]))
                        r_new = solve_lp(c_tmp)
                        best_x[2::3] = r_new
                        best_sum = np.sum(r_new)
            except Exception:
                pass

    # Phase 5: Uniform Expansion Push
    if best_x is not None:
        x_exp = best_x.copy()
        x_exp[2::3] *= 1.015
        for i in range(N):
            r = x_exp[3 * i + 2]
            x_exp[3 * i] = np.clip(x_exp[3 * i], r, 1.0 - r)
            x_exp[3 * i + 1] = np.clip(x_exp[3 * i + 1], r, 1.0 - r)
            
        try:
            res = minimize(objective, x_exp, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                vals = constraint_fun(res.x)
                if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                    best_x = res.x.copy()
                    best_sum = -res.fun
        except Exception:
            pass

    # Fallback (should rarely be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[2::3] = 0.08
        best_x[0::3] = np.tile(np.linspace(0.12, 0.88, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.12, 0.88, 6), 5)[:N]

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]

    # Final strict validation & numerical repair
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
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])

    return centers, radii, float(np.sum(radii))
