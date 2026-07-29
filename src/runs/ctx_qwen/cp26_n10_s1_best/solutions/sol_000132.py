# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000118 (state 224f6ad6) state=ce7773d8 sum of radii=2.629964 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    # Boundary constraints
    c = np.concatenate([
        xs - rs, 1.0 - xs - rs,
        ys - rs, 1.0 - ys - rs
    ])
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, dx[TRIL]**2 + dy[TRIL]**2 - dr[TRIL]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

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
            
    bounds_lp = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_hex_init(scale=1.0, angle=0.0):
    """Generates a hexagonal lattice initialization with optional rotation and scaling."""
    pts = []
    r0 = 0.095 * scale
    y = r0
    row = 0
    while len(pts) < N + 5:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 + r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N + 5])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    # Center and uniformly scale to fit inside [0.1, 0.9]
    pts -= pts.mean(axis=0)
    max_d = np.abs(pts).max()
    if max_d > 1e-5:
        pts *= 0.4 / max_d
    pts += 0.5
    
    return pts[:N]

def make_grid_init():
    """Generates a perturbed grid initialization."""
    pts = []
    for i in range(6):
        for j in range(5):
            if len(pts) < N:
                pts.append([0.1 + j * 0.2, 0.1 + i * 0.18])
    pts = np.array(pts[:N])
    pts += np.random.uniform(-0.01, 0.01, pts.shape)
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
    
    # Hexagonal lattices with varied scales and rotations
    for sc in [0.9, 1.0, 1.1, 1.2]:
        for ang in np.linspace(-0.4, 0.4, 9):
            inits.append(make_hex_init(scale=sc, angle=ang))
            
    # Grid initialization
    inits.append(make_grid_init())
    
    # Random starts with force-directed pre-spreading
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Quick repulsion to avoid initial overlaps
        for _ in range(100):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.2 and d > 1e-6:
                        rep = 0.005 / (d**2 + 0.001)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
                for dim in range(2):
                    if pts[i, dim] < 0.15: f[i, dim] += 0.03
                    elif pts[i, dim] > 0.85: f[i, dim] -= 0.03
            pts += f * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 2: Multi-start Optimization
    for pts in inits:
        r_lp = solve_lp_radii(pts)
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = np.maximum(r_lp * 0.995, 0.01)
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
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
        for cycle in range(50):
            x0 = best_x.copy()
            # Progressive deflation to create slack
            shrink = 0.90 + 0.08 * (cycle / 50.0)
            x0[2::3] *= shrink
            
            noise_scale = 0.002 / (cycle + 1)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x0[2::3] = r_new * 0.998
                        x0 = project_to_feasible(x0)
                        s = np.sum(x0[2::3])
                        if s > best_sum:
                            best_sum = s
                            best_x = x0.copy()
            except Exception:
                pass

    # Phase 4: High-Precision Polish
    if best_x is not None:
        for _ in range(5):
            x_polish = best_x.copy()
            x_polish[2::3] *= 0.99
            x_polish = project_to_feasible(x_polish)
            try:
                res = minimize(objective, x_polish, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_final = solve_lp_radii(c_opt)
                    if r_final is not None:
                        x_polish[2::3] = r_final * 0.999
                        x_polish = project_to_feasible(x_polish)
                        s = np.sum(x_polish[2::3])
                        if s > best_sum:
                            best_sum = s
                            best_x = x_polish.copy()
            except Exception:
                break

    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
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
