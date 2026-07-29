# sol_000197 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000186 (state c3cfc6eb) state=dbe786fd sum of radii=2.619132 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    m = n + n * (n - 1) // 2
    A = np.zeros((m, n))
    b = np.zeros(m)
    idx = 0
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
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
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 1e-4)

def constraints_slsqp(x):
    """Inequality constraints for SLSQP: g(x) >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def objective_slsqp(x):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_hex_init(r0, angle):
    """Generates a rotated hexagonal lattice initialization."""
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
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
    return pts[:N]

def run_packing():
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_slsqp}
    
    best_sum = -1.0
    best_x = None

    # Phase 1: Diverse Initial Configurations & SLSQP Polish
    inits = []
    for ang in np.linspace(-0.35, 0.35, 11):
        inits.append(make_hex_init(0.092, ang))
    for s in range(8):
        np.random.seed(s + 100)
        inits.append(np.random.uniform(0.15, 0.85, (N, 2)))

    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = r_lp * 0.995
        x0 = project_to_feasible(x0)
        try:
            res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                curr_sum = np.sum(r_new)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
                    best_x[2::3] = r_new
        except Exception:
            continue

    # Phase 2: Simulated Annealing on Centers (uses exact LP evaluation)
    if best_x is not None:
        curr_c = np.column_stack((best_x[0::3], best_x[1::3]))
        curr_sum = best_sum
        T = 0.008
        decay = 0.995
        
        for step in range(900):
            noise = np.random.normal(0, 0.007, curr_c.shape)
            new_c = np.clip(curr_c + noise, 0.005, 0.995)
            
            new_r = solve_lp_radii(new_c)
            new_sum = np.sum(new_r)
            
            delta = new_sum - curr_sum
            if delta > 0 or (T > 1e-6 and np.random.rand() < np.exp(delta / T)):
                curr_c = new_c
                curr_sum = new_sum
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x[0::3] = curr_c[:, 0]
                    best_x[1::3] = curr_c[:, 1]
                    best_x[2::3] = new_r
            T *= decay

    # Phase 3: Aggressive Topology Search (Deflation & Repositioning)
    if best_x is not None:
        for cyc in range(70):
            x0 = best_x.copy()
            noise = 0.0025 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 8)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.80
            
            x0 = project_to_feasible(x0)
            try:
                res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    s = np.sum(r_new)
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
                        best_x[2::3] = r_new
            except Exception:
                pass

    # Phase 4: Global Rotation Search
    if best_x is not None:
        bc = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-15, 15, 13):
            ang = np.deg2rad(ang_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            rot_c = (bc - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
            rot_c = np.clip(rot_c, 0.01, 0.99)
            r_rot = solve_lp_radii(rot_c)
            
            x_rot = np.zeros(3 * N)
            x_rot[0::3] = rot_c[:, 0]
            x_rot[1::3] = rot_c[:, 1]
            x_rot[2::3] = r_rot * 0.99
            try:
                res = minimize(objective_slsqp, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_r = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_r = solve_lp_radii(c_r)
                    if np.sum(r_r) > best_sum:
                        best_sum = np.sum(r_r)
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_r[:, 0]
                        best_x[1::3] = c_r[:, 1]
                        best_x[2::3] = r_r
            except Exception:
                pass

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
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        radii *= 0.9998
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(np.sum(radii))
