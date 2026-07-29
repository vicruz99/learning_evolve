# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000069 (state 13ab459c) state=e35be59b sum of radii=2.602672 correctness=1.0
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
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
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
        bound = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = bound
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], 
                            centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def make_staggered_init():
    """Generate a 5-6-5-6-4 staggered grid initialization."""
    pts = []
    rows = [5, 6, 5, 6, 4]
    y_spacing = 1.0 / 5.3
    y = y_spacing
    for r_idx, count in enumerate(rows):
        x_spacing = 1.0 / (count + 1)
        x = x_spacing
        shift = 0.0 if r_idx % 2 == 0 else x_spacing * 0.5
        for _ in range(count):
            pts.append([x + shift, y])
            x += x_spacing
        y += y_spacing
    return np.clip(np.array(pts[:N]), 0.02, 0.98)

def make_hex_init(r0, angle):
    """Generate a hexagonal lattice initialization with optional rotation."""
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
        c_a, s_a = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_a, -s_a], [s_a, c_a]]) + 0.5
        
    pts += np.random.normal(0, 0.003, pts.shape)
    return np.clip(pts[:N], 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    inits = []
    
    # 1. Staggered grids
    inits.append(make_staggered_init())
    
    # 2. Hexagonal lattices with varied scales and rotations
    for seed in range(25):
        r0 = 0.075 + seed * 0.004
        ang = seed * 0.18
        inits.append(make_hex_init(r0, ang))
        
    # 3. Force-directed spreads from random starts
    for _ in range(10):
        pts = np.random.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[j] - pts[i]
                    dist = np.hypot(diff[0], diff[1])
                    if dist < 0.25 and dist > 1e-5:
                        force = 0.0008 / (dist**2 + 1e-4)
                        f[i] -= diff * force / dist
                        f[j] += diff * force / dist
            pts += f * 0.03
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 1: Multi-start Optimization
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        # Initialize radii via LP for maximum head-start
        r_lp = solve_lp_radii(pts)
        if r_lp is not None:
            x0[2::3] = np.maximum(r_lp * 0.992, 0.005)
        else:
            x0[2::3] = 0.07
            
        # Project strictly feasible
        for i in range(N):
            r = x0[3 * i + 2]
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                s = -res.fun
                vals = constraint_fun(res.x)
                if np.min(vals) >= -1e-6 and s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Alternating Refinement (LP Radii <-> SLSQP Positions)
    if best_x is not None:
        for cycle in range(40):
            x0 = best_x.copy()
            
            # Strategy alternates between full deflation and subset deflation
            if cycle % 3 == 0:
                x0[2::3] *= 0.94
            else:
                subset = np.random.choice(N, size=N // 4, replace=False)
                x0[3 * subset + 2] *= 0.65
                
            # Perturb centers to escape local minima
            noise_scale = 0.0025 / (cycle + 1)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            # Project back to strict bounds
            for i in range(N):
                r = max(0.005, x0[3 * i + 2])
                x0[3 * i + 2] = r
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    s = -res.fun
                    if s > best_sum:
                        best_x = res.x.copy()
                        best_sum = s
                        
                    # LP snap: optimally scale radii for current centers
                    c_tmp = np.column_stack((best_x[0::3], best_x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        best_x[2::3] = r_new * 0.997
                        best_sum = np.sum(best_x[2::3])
            except Exception:
                pass

    # Fallback (should rarely be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.12, 0.88, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.12, 0.88, 6), 5)[:N]
        best_x[2::3] = 0.08
        
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
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9995
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
            
    return centers, radii, float(np.sum(radii))
