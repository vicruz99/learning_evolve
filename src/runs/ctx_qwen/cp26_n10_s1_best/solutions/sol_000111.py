# sol_000111 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000069 (state 13ab459c) state=3d33d0a6 sum of radii=2.627905 correctness=1.0
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
        b_ub[idx] = max(0.0, bound)
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
    return np.full(n, 0.01)

def make_hex_init(r0, angle=0.0):
    """Generate hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        if y > 1.0 + r0: break
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c_a, s_a = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_a, -s_a], [s_a, c_a]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_spread_init(seed, steps=300):
    """Force-directed spread initialization."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(steps):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[j] - pts[i]
                d = np.linalg.norm(diff)
                if d < 0.3 and d > 1e-5:
                    force = 0.0008 / (d**2 + 0.001)
                    f[i] -= diff * force / d
                    f[j] += diff * force / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.03
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.03
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def project_to_bounds(x0):
    """Ensure optimization vector strictly respects bounds."""
    x0 = x0.copy()
    for i in range(N):
        r = max(1e-7, x0[3 * i + 2])
        x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
        x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
        x0[3 * i + 2] = r
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattices with varied spacing and rotation
    for r0 in np.linspace(0.075, 0.105, 7):
        for ang in np.linspace(-0.35, 0.35, 9):
            inits.append(make_hex_init(r0, ang))
            
    # Force-directed spreads
    for s in range(8):
        inits.append(force_spread_init(s))
        
    # Structured grids
    for _ in range(4):
        pts = np.zeros((N, 2))
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx < N:
                    pts[idx] = [0.1 + c * 0.18 + np.random.uniform(-0.01, 0.01), 
                                0.1 + r * 0.16 + np.random.uniform(-0.01, 0.01)]
                    idx += 1
        inits.append(pts)

    # Phase 2: Multi-start Optimization with LP initialization
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        r_lp = solve_lp_radii(pts)
        x0[2::3] = np.maximum(r_lp * 0.995, 0.005)
        x0 = project_to_bounds(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                s = -res.fun
                vals = constraint_fun(res.x)
                if np.min(vals) >= -1e-6 and s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Alternating Refinement & Topology Search
    if best_x is not None:
        for cycle in range(40):
            # Deflate radii to allow centers to move
            noise_scale = 0.003 * (0.92 ** cycle)
            x0 = best_x.copy()
            x0[2::3] *= 0.96
            
            # Perturb centers
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            # Topology swap: randomly exchange positions of 2 circles
            if cycle % 5 == 0:
                i_swap, j_swap = np.random.choice(N, 2, replace=False)
                x0[3*i_swap], x0[3*j_swap] = x0[3*j_swap], x0[3*i_swap]
                x0[3*i_swap+1], x0[3*j_swap+1] = x0[3*j_swap+1], x0[3*i_swap+1]
                
            x0 = project_to_bounds(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass
                
            # LP refinement on current centers
            c_tmp = np.column_stack((best_x[0::3], best_x[1::3]))
            r_new = solve_lp_radii(c_tmp)
            if r_new is not None:
                best_x[2::3] = r_new * 0.998
                best_sum = np.sum(best_x[2::3])
                
                # Re-optimize positions with slightly deflated radii
                x0_fix = best_x.copy()
                x0_fix[2::3] *= 0.98
                x0_fix = project_to_bounds(x0_fix)
                try:
                    res = minimize(objective, x0_fix, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun):
                        s = -res.fun
                        vals = constraint_fun(res.x)
                        if np.min(vals) >= -1e-6 and s > best_sum:
                            best_sum = s
                            best_x = res.x.copy()
                except Exception:
                    pass

    # Fallback initialization
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.08
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final strict validation repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or \
               centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False; break
                if not valid: break
        if valid: break
        radii *= 0.9995
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
            
    return centers, radii, float(np.sum(radii))
