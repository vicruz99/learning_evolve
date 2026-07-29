# sol_000177 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000167 (state 1af0cc64) state=2741b787 sum of radii=2.629585 correctness=1.0
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
    """Returns all inequality constraints >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Overlap constraints: dist >= r_i + r_j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-8, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)] * N

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
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-8, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

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
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, 2)])
    return np.clip(pts[:N], 0.05, 0.95)

def make_grid_init():
    """Generates a grid-based initialization with jitter."""
    pts = np.zeros((N, 2))
    idx = 0
    xs = np.linspace(0.12, 0.88, 5)
    ys = np.linspace(0.12, 0.88, 5)
    for yi in ys:
        for xi in xs:
            if idx < N:
                pts[idx] = [xi, yi]
                idx += 1
    while idx < N:
        pts[idx] = np.random.uniform(0.2, 0.8, 2)
        idx += 1
    pts += np.random.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    # Hexagonal lattices with scale and rotation variations
    for scale in [0.92, 0.96, 1.0, 1.04, 1.08]:
        for ang in np.linspace(-0.45, 0.45, 11):
            inits.append(make_hex_init(0.09 * scale, ang))
    # Grid-based starts
    for _ in range(8):
        inits.append(make_grid_init())
    # Force-directed random starts
    for s in range(8):
        np.random.seed(s * 31 + 13)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(60):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1])
                    if d < 0.28 and d > 1e-4:
                        rep = 0.006 / (d**2 + 0.001)
                        dx = pts[j, 0] - pts[i, 0]
                        dy = pts[j, 1] - pts[i, 1]
                        f[i, 0] -= dx * rep / d
                        f[i, 1] -= dy * rep / d
                        f[j, 0] += dx * rep / d
                        f[j, 1] += dy * rep / d
            pts += f * 0.04
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 2: Multi-start Optimization with Alternating LP/SLSQP
    for c0 in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = solve_lp_radii(c0) * 0.97
        x0 = project_to_feasible(x0)
        
        try:
            # Primary SLSQP run
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if np.isnan(res.fun):
                continue
                
            x_tmp = res.x.copy()
            # Alternate: LP on optimized centers, then SLSQP again
            c_tmp = np.column_stack((x_tmp[0::3], x_tmp[1::3]))
            r_lp = solve_lp_radii(c_tmp)
            x_tmp[2::3] = r_lp
            x_tmp = project_to_feasible(x_tmp)
            
            res2 = minimize(objective, x_tmp, method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res2.fun):
                x_final = res2.x.copy()
                c_final = np.column_stack((x_final[0::3], x_final[1::3]))
                r_final = solve_lp_radii(c_final)
                x_final[2::3] = r_final
                x_final = project_to_feasible(x_final)
                
                s_val = np.sum(r_final)
                if s_val > best_sum and np.min(constraints(x_final)) >= -1e-4:
                    best_sum = s_val
                    best_x = x_final.copy()
        except Exception:
            continue
            
    # Phase 3: Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cyc in range(50):
            x0 = best_x.copy()
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(4, 9)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.82
            
            # Perturb centers with cooling noise
            noise_scale = 0.0025 * (0.94 ** cyc)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 9000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(c_p)
                    x_p = res.x.copy()
                    x_p[2::3] = r_p
                    x_p = project_to_feasible(x_p)
                    
                    if np.min(constraints(x_p)) >= -1e-4:
                        s_p = np.sum(r_p)
                        if s_p > best_sum:
                            best_sum = s_p
                            best_x = x_p.copy()
            except Exception:
                pass
                
    # Phase 4: High-Precision Polish
    if best_x is not None:
        for _ in range(3):
            res = minimize(objective, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                best_x = res.x.copy()
                c_pol = np.column_stack((best_x[0::3], best_x[1::3]))
                r_pol = solve_lp_radii(c_pol)
                best_x[2::3] = r_pol
                best_x = project_to_feasible(best_x)
                
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
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
