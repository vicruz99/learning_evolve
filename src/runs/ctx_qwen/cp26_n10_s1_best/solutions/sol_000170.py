# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000161 (state 2acf7cee) state=dbfe0634 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)
TRIU_IDX = np.triu_indices(N, 1)

def objective_func(x):
    """Objective: maximize sum of radii."""
    return -np.sum(x[2::3])

def constraints_slsqp(x):
    """Inequality constraints for SLSQP: g(x) >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    A = np.zeros((n + n*(n-1)//2, n))
    b = np.zeros(n + n*(n-1)//2)
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
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def center_penalty(c_flat, radii):
    """Smooth penalty function for centers given fixed radii."""
    c = c_flat.reshape(-1, 2)
    p = 0.0
    x, y, r = c[:, 0], c[:, 1], radii
    p += np.sum(np.maximum(r - x, 0.0)**2)
    p += np.sum(np.maximum(x - (1.0 - r), 0.0)**2)
    p += np.sum(np.maximum(r - y, 0.0)**2)
    p += np.sum(np.maximum(y - (1.0 - r), 0.0)**2)
    dx = c[:, 0, None] - c[:, 0]
    dy = c[:, 1, None] - c[:, 1]
    d = np.hypot(dx, dy)
    gap = d - r[:, None] - r[None, :]
    p += np.sum(np.maximum(-gap[TRIU_IDX], 0.0)**2)
    return p

def opt_centers_lbfgs(centers, radii):
    """Optimize centers for fixed radii using L-BFGS-B."""
    bounds = [(0.001, 0.999)] * (2 * N)
    res = minimize(center_penalty, centers.flatten(), args=(radii,),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-15})
    return res.x.reshape(-1, 2)

def make_init_hex(r0, angle):
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
        ca, sa = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_slsqp}
    best_sum = -1.0
    best_x = None

    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    for ang in np.linspace(-0.4, 0.4, 21):
        inits.append(make_init_hex(0.095, ang))
    for s in range(12):
        np.random.seed(s)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.25 and d > 1e-5:
                        rep = 0.01 / (d**2 + 0.001)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
                for dim in range(2):
                    if pts[i, dim] < 0.1: f[i, dim] += 0.05
                    elif pts[i, dim] > 0.9: f[i, dim] -= 0.05
            pts += f * 0.05
            pts = np.clip(pts, 0.03, 0.97)
        inits.append(pts)

    # Phase 2: Alternating Optimization + SLSQP Polish
    for pts in inits:
        c = pts.copy()
        r = solve_lp_radii(c)
        for _ in range(20):
            r = solve_lp_radii(c)
            c = opt_centers_lbfgs(c, r)
        x0 = np.zeros(3 * N)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = np.maximum(r * 0.995, 1e-6)
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum and np.min(constraints_slsqp(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Aggressive Topology Search (Deflation & Repositioning)
    if best_x is not None:
        for cyc in range(100):
            x0 = best_x.copy()
            noise = 0.0025 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            subset = np.random.choice(N, size=max(4, N // 3), replace=False)
            x0[subset * 3 + 2] *= 0.80
            for i in range(N):
                r = max(1e-6, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
            try:
                res = minimize(objective_func, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(c_p)
                    if np.sum(r_p) > best_sum:
                        x_p = res.x.copy()
                        x_p[2::3] = r_p
                        for i in range(N):
                            x_p[3 * i] = np.clip(x_p[3 * i], r_p[i], 1.0 - r_p[i])
                            x_p[3 * i + 1] = np.clip(x_p[3 * i + 1], r_p[i], 1.0 - r_p[i])
                        if np.min(constraints_slsqp(x_p)) >= -1e-6:
                            best_sum = np.sum(r_p)
                            best_x = x_p.copy()
            except Exception:
                pass

    # Phase 4: Rotation Search & Uniform Expansion
    if best_x is not None:
        bc = np.column_stack((best_x[0::3], best_x[1::3]))
        # Rotation search
        for ang_deg in np.linspace(-15, 15, 17):
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
                res = minimize(objective_func, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13})
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
        # Uniform expansion attempt
        x_exp = best_x.copy()
        x_exp[2::3] *= 1.02
        for i in range(N):
            r = x_exp[3 * i + 2]
            x_exp[3 * i] = np.clip(x_exp[3 * i], r, 1.0 - r)
            x_exp[3 * i + 1] = np.clip(x_exp[3 * i + 1], r, 1.0 - r)
        try:
            res = minimize(objective_func, x_exp, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14})
            if not np.isnan(res.fun):
                c_exp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_exp = solve_lp_radii(c_exp)
                if np.sum(r_exp) > best_sum:
                    best_sum = np.sum(r_exp)
                    best_x = np.zeros(3 * N)
                    best_x[0::3] = c_exp[:, 0]
                    best_x[1::3] = c_exp[:, 1]
                    best_x[2::3] = r_exp
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
