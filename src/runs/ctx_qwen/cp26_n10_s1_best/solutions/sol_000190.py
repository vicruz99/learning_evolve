# sol_000190 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000170 (state dbfe0634) state=6369cf88 sum of radii=2.627694 correctness=1.0
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
    """Inequality constraints for SLSQP: g(x) >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    # Boundary constraints
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    # Overlap constraints using hypot for stable unit-gradient at contact
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N

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
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(r0, angle):
    """Generates a rotated hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 5:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    pts = np.array(pts[:N + 5])
    if angle != 0.0:
        ca, sa = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_spread(centers, steps=200):
    """Force-directed relaxation to spread points evenly and push to boundaries."""
    pts = centers.copy()
    for s in range(steps):
        f = np.zeros_like(pts)
        lr = 0.02 * (1.0 - s / steps)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-5:
                    rep = 0.01 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.1: f[i, dim] += 0.05
                elif pts[i, dim] > 0.9: f[i, dim] -= 0.05
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    best_sum = -1.0
    best_x = None

    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    for r0 in [0.088, 0.092, 0.096]:
        for ang in np.linspace(-0.35, 0.35, 11):
            inits.append(make_hex_init(r0, ang))
    for s in range(15):
        np.random.seed(s)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        inits.append(force_spread(pts, steps=150))

    # Phase 2: Multi-start Optimization with LP Warm-start
    for pts in inits:
        r = solve_lp_radii(pts)
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = np.maximum(r * 0.99, 1e-6)
        # Project to strictly feasible bounds
        for i in range(N):
            ri = x0[3 * i + 2]
            x0[3 * i] = np.clip(x0[3 * i], ri, 1.0 - ri)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], ri, 1.0 - ri)
        # Break symmetries
        x0 += np.random.normal(0, 1e-4, x0.shape)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Topology Escape (Deflation & Perturbation)
    if best_x is not None:
        for cyc in range(80):
            x0 = best_x.copy()
            noise = 0.0025 * (0.92 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 8)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.85
            
            for i in range(N):
                ri = max(1e-6, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], ri, 1.0 - ri)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], ri, 1.0 - ri)
                x0[3 * i + 2] = ri

            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(c_p)
                    if np.sum(r_p) > best_sum:
                        x_p = res.x.copy()
                        x_p[2::3] = r_p
                        for i in range(N):
                            ri = r_p[i]
                            x_p[3 * i] = np.clip(x_p[3 * i], ri, 1.0 - ri)
                            x_p[3 * i + 1] = np.clip(x_p[3 * i + 1], ri, 1.0 - ri)
                        if np.min(constraints(x_p)) >= -1e-6:
                            best_sum = np.sum(r_p)
                            best_x = x_p.copy()
            except Exception:
                pass

        # Phase 4: Global Rotation Search
        bc = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-10, 10, 11):
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
                res = minimize(objective, x_rot, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
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
