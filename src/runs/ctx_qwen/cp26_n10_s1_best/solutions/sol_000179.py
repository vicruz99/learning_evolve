# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000167 (state 1af0cc64) state=3dcf3ee1 sum of radii=2.629754 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-9, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-9, 0.5)] * N

def constraint_func(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    num_ineq = n + n * (n - 1) // 2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A[idx, i] = 1.0
        b[idx] = max(0.0, lim)
        idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, d)
            idx += 1
    bounds = [(0.0, None)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(scale=1.0, angle=0.0):
    """Generates a hexagonal lattice initialization tailored for 26 circles."""
    pts = []
    r = 0.1 * scale
    y = r
    counts = [5, 6, 5, 6, 4]
    for cnt in counts:
        row_w = (cnt - 1) * 2.0 * r
        x_s = (1.0 - row_w) / 2.0
        for k in range(cnt):
            pts.append([x_s + k * 2.0 * r, y])
        y += np.sqrt(3.0) * r
    pts = np.array(pts[:N])
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    return np.clip(pts, 0.01, 0.99)

def make_random_init(seed):
    """Force-directed random initialization."""
    np.random.seed(seed)
    pts = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(200):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-5:
                    rep = 0.015 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.08: f[i, dim] += 0.06
                elif pts[i, dim] > 0.92: f[i, dim] -= 0.06
        pts += f * 0.05
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multi-start SLSQP Optimization
    inits = []
    for ang in np.linspace(-0.3, 0.3, 11):
        inits.append(make_hex_init(scale=1.0, angle=ang))
        inits.append(make_hex_init(scale=1.05, angle=ang))
    for s in range(15):
        inits.append(make_random_init(s * 13 + 7))
        
    for c0 in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        r0 = solve_lp_radii(c0)
        x0[2::3] = r0 * 0.99
        for i in range(N):
            r = x0[3*i+2]
            x0[3*i] = np.clip(x0[3*i], r, 1.0-r)
            x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0-r)
            
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 30000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                x_tmp = res.x.copy()
                c_tmp = np.column_stack((x_tmp[0::3], x_tmp[1::3]))
                r_lp = solve_lp_radii(c_tmp)
                x_tmp[2::3] = r_lp
                s_val = np.sum(r_lp)
                if s_val > best_sum and np.min(constraint_func(x_tmp)) >= -1e-5:
                    best_sum = s_val
                    best_x = x_tmp.copy()
        except Exception:
            continue

    # Phase 2: Simulated Annealing on Centers with LP Evaluation
    if best_x is not None:
        curr_c = np.column_stack((best_x[0::3], best_x[1::3]))
        best_c = curr_c.copy()
        best_v = best_sum
        
        temp = 0.015
        for step in range(3000):
            temp *= 0.9985
            k = np.random.randint(2, 6)
            idx = np.random.choice(N, k, replace=False)
            new_c = curr_c.copy()
            new_c[idx] += np.random.normal(0, temp, (k, 2))
            new_c = np.clip(new_c, 1e-5, 1.0 - 1e-5)
            
            v_new = np.sum(solve_lp_radii(new_c))
            delta = v_new - best_v
            
            if delta > 0 or (temp > 1e-10 and np.random.rand() < np.exp(delta / (temp + 1e-12))):
                curr_c = new_c
                if v_new > best_v:
                    best_v = v_new
                    best_c = new_c.copy()
                    
        if best_v > best_sum:
            best_sum = best_v
            best_x = np.zeros(3 * N)
            best_x[0::3] = best_c[:, 0]
            best_x[1::3] = best_c[:, 1]
            best_x[2::3] = solve_lp_radii(best_c)

    # Phase 3: Coordinate-wise Fine Tuning
    if best_x is not None:
        curr_c = np.column_stack((best_x[0::3], best_x[1::3]))
        improved = True
        while improved:
            improved = False
            for i in range(N):
                base_s = np.sum(best_x[2::3])
                for dx in [-0.003, -0.001, 0.001, 0.003]:
                    for dy in [-0.003, -0.001, 0.001, 0.003]:
                        nc = curr_c.copy()
                        nc[i, 0] = np.clip(curr_c[i, 0] + dx, 0.001, 0.999)
                        nc[i, 1] = np.clip(curr_c[i, 1] + dy, 0.001, 0.999)
                        rn = solve_lp_radii(nc)
                        sn = np.sum(rn)
                        if sn > base_s + 1e-8:
                            curr_c = nc
                            best_x[0::3] = curr_c[:, 0]
                            best_x[1::3] = curr_c[:, 1]
                            best_x[2::3] = rn
                            best_sum = sn
                            improved = True
                            break
                    if improved:
                        break

    # Final strict validation and minimal repair
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    for _ in range(50):
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
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
