# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000150 (state e23cc911) state=9bd4a86f sum of radii=2.619299 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def get_bounds():
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def solve_lp_radii(centers):
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
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def constraints_slsqp(x):
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
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
    return -np.sum(x[2::3])

def project_to_feasible(x):
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_hex_init(r0, angle):
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
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def make_corner_init():
    pts = np.zeros((N, 2))
    r = 0.09
    pts[0] = [r, r]
    pts[1] = [1 - r, r]
    pts[2] = [r, 1 - r]
    pts[3] = [1 - r, 1 - r]
    pts[4] = [0.5, r]
    pts[5] = [0.5, 1 - r]
    pts[6] = [r, 0.5]
    pts[7] = [1 - r, 0.5]
    idx = 8
    y = r + 0.05
    row = 0
    while idx < N and y + r <= 1.0:
        x_start = r + 0.06 if row % 2 == 0 else r + 0.18
        x = x_start
        while idx < N and x + r <= 1.0:
            pts[idx] = [x, y]
            idx += 1
            x += 0.22
        y += 0.15
        row += 1
    for i in range(idx, N):
        pts[i] = np.random.uniform(0.1, 0.9, 2)
    pts += np.random.uniform(-0.005, 0.005, pts.shape)
    return np.clip(pts, 0.05, 0.95)

def run_packing():
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints_slsqp}
    best_sum = -1.0
    best_x = None

    # Phase 1: Diverse Initial Configurations
    inits = []
    for ang in np.linspace(-0.3, 0.3, 13):
        inits.append(make_hex_init(0.092, ang))
        inits.append(make_hex_init(0.098, ang))
    for _ in range(6):
        inits.append(make_corner_init())
    for s in range(8):
        np.random.seed(s + 2000)
        inits.append(np.random.uniform(0.15, 0.85, (N, 2)))

    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        r_lp = solve_lp_radii(pts)
        x0[2::3] = r_lp * 0.995
        x0 = project_to_feasible(x0)
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                x_tmp = res.x.copy()
                x_tmp[2::3] = r_new
                s_val = np.sum(r_new)
                if s_val > best_sum and np.min(constraints_slsqp(x_tmp)) >= -1e-6:
                    best_sum = s_val
                    best_x = x_tmp.copy()
        except Exception:
            pass

    # Phase 2: Aggressive Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cyc in range(80):
            x0 = best_x.copy()
            shrink = 0.80 + 0.20 * (cyc / 80.0)
            x0[2::3] *= shrink
            noise = 0.002 * (0.90 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate random subset to break rigid contact networks
            k = np.random.randint(4, 9)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.85
            
            x0 = project_to_feasible(x0)
            try:
                res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    x_tmp = res.x.copy()
                    x_tmp[2::3] = r_new
                    x_tmp = project_to_feasible(x_tmp)
                    s_val = np.sum(r_new)
                    if s_val > best_sum and np.min(constraints_slsqp(x_tmp)) >= -1e-5:
                        best_sum = s_val
                        best_x = x_tmp.copy()
            except Exception:
                pass

    # Phase 3: Coordinate-wise Fine Tuning
    if best_x is not None:
        centers = np.column_stack((best_x[0::3], best_x[1::3]))
        radii = best_x[2::3].copy()
        for _ in range(4):
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    best_local = np.sum(radii)
                    for dx in [-0.004, -0.0015, 0.0015, 0.004]:
                        for dy in [-0.004, -0.0015, 0.0015, 0.004]:
                            if dx == 0 and dy == 0:
                                continue
                            new_c = centers.copy()
                            new_c[i, 0] = np.clip(new_c[i, 0] + dx, 0.005, 0.995)
                            new_c[i, 1] = np.clip(new_c[i, 1] + dy, 0.005, 0.995)
                            r_cand = solve_lp_radii(new_c)
                            s_cand = np.sum(r_cand)
                            if s_cand > best_local + 1e-7:
                                centers = new_c
                                radii = r_cand
                                best_local = s_cand
                                best_x[0::3] = centers[:, 0]
                                best_x[1::3] = centers[:, 1]
                                best_x[2::3] = radii
                                best_sum = s_cand
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                # SLSQP polish after coordinate descent
                if not improved:
                    x_pol = np.zeros(3 * N)
                    x_pol[0::3] = centers[:, 0]
                    x_pol[1::3] = centers[:, 1]
                    x_pol[2::3] = radii
                    try:
                        res_p = minimize(objective_func, x_pol, method='SLSQP', bounds=bounds, constraints=cons,
                                         options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                        if not np.isnan(res_p.fun):
                            c_p = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                            r_p = solve_lp_radii(c_p)
                            if np.sum(r_p) > best_sum + 1e-7:
                                centers = c_p
                                radii = r_p
                                best_x[0::3] = centers[:, 0]
                                best_x[1::3] = centers[:, 1]
                                best_x[2::3] = radii
                                best_sum = np.sum(r_p)
                                improved = True
                    except Exception:
                        pass

    # Final extraction and repair
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final LP squeeze
    final_r = solve_lp_radii(centers)
    if np.sum(final_r) > np.sum(radii) - 1e-7:
        radii = final_r.copy()

    # Strict validation repair
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
