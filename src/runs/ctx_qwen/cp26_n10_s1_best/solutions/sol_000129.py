# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000118 (state 224f6ad6) state=644ff0b9 sum of radii=2.627905 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL = np.tril_indices(N, -1)

def get_bounds():
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def solve_lp_radii(centers):
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
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
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.06)

def project_to_feasible(x):
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def force_resolve(centers, radii, steps=50):
    pts = centers.copy()
    r = radii.copy()
    for _ in range(steps):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-6:
                    push = (min_d - d) * 0.5 / d
                    f[i] -= dx * push
                    f[j] += dx * push
            for dim in range(2):
                if pts[i, dim] < r[i]: f[i, dim] += (r[i] - pts[i, dim]) * 0.5
                if pts[i, dim] > 1.0 - r[i]: f[i, dim] -= (pts[i, dim] - (1.0 - r[i])) * 0.5
        pts += f
        pts = np.clip(pts, 0.0, 1.0)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    
    def objective(x):
        return -np.sum(x[2::3])

    def constraints(x):
        xs, ys, rs = x[0::3], x[1::3], x[2::3]
        c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dr = rs[:, None] + rs[None, :]
        c = np.concatenate([c, dx[TRIL]**2 + dy[TRIL]**2 - dr[TRIL]**2])
        return c

    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    
    # 1. Rotated Hexagonal Lattices
    for scale in [0.95, 1.0, 1.05, 1.1]:
        for ang in np.linspace(-0.25, 0.25, 9):
            pts = []
            r0 = 0.09 * scale
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
            c, s = np.cos(ang), np.sin(ang)
            pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            pts += np.random.uniform(-0.002, 0.002, pts.shape)
            inits.append(pts)
            
    # 2. Force-directed layouts pushing to boundaries
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        radii = np.full(N, 0.05)
        for step in range(300):
            f = np.zeros_like(pts)
            lr = 0.04 * (1.0 - step / 300)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.25 and d > 1e-5:
                        rep = 0.01 / (d**2 + 0.001)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
                for dim in range(2):
                    if pts[i, dim] < 0.15: f[i, dim] += 0.06
                    elif pts[i, dim] > 0.85: f[i, dim] -= 0.06
            pts += f * lr
            pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)

    # Phase 2: Multi-start Optimization with LP Initialization
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        r_lp = solve_lp_radii(pts)
        x0[2::3] = np.maximum(r_lp * 0.995, 0.005)
        x0 = project_to_feasible(x0)
        
        # Quick force resolution to remove major overlaps before SLSQP
        centers_tmp = np.column_stack((x0[0::3], x0[1::3]))
        centers_tmp = force_resolve(centers_tmp, x0[2::3], steps=30)
        x0[0::3] = centers_tmp[:, 0]
        x0[1::3] = centers_tmp[:, 1]
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                s = -res.fun
                vals = constraints(res.x)
                if np.min(vals) >= -1e-5 and s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Aggressive Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cycle in range(50):
            # Perturb and deflate random subset to break rigid contacts
            noise_scale = 0.0015 / (cycle + 1)
            x0 = best_x + np.random.normal(0, noise_scale, 3 * N)
            
            subset = np.random.choice(N, size=N // 3, replace=False)
            x0[subset * 3 + 2] *= 0.80
            
            x0 = project_to_feasible(x0)
            
            # Re-resolve overlaps with forces
            centers_tmp = np.column_stack((x0[0::3], x0[1::3]))
            centers_tmp = force_resolve(centers_tmp, x0[2::3], steps=20)
            x0[0::3] = centers_tmp[:, 0]
            x0[1::3] = centers_tmp[:, 1]
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    # LP refinement on new centers to snap radii to theoretical max
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x0[2::3] = r_new * 0.998
                        x0 = project_to_feasible(x0)
                        s = np.sum(x0[2::3])
                        if s > best_sum:
                            best_sum = s
                            best_x = x0.copy()
                            
                            # Final SLSQP polish on improved state
                            res2 = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                            if not np.isnan(res2.fun) and -res2.fun > best_sum:
                                best_x = res2.x.copy()
                                best_sum = -res2.fun
            except Exception:
                pass

    # Fallback initialization
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        
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
