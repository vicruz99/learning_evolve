# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000143 (state f27c5ca1) state=8c767842 sum of radii=2.630713 correctness=1.0
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
    """Returns all inequality constraints g(x) >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
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
    A = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.01)

def project_x(x0):
    """Project variables to strictly satisfy bounds."""
    x0 = x0.copy()
    for i in range(N):
        r = max(1e-7, x0[3 * i + 2])
        x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
        x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
        x0[3 * i + 2] = r
    return x0

def make_hex_init(r0, angle=0.0):
    """Generates a hexagonal lattice initialization with optional rotation."""
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
        
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c_a, s_a = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_a, -s_a], [s_a, c_a]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
    return pts[:N]

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(200):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.linalg.norm(dx)
                if d < 0.25 and d > 1e-6:
                    rep = 0.006 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.04
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.04
        pts += f * 0.04
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def coordinate_descent(centers, radii, steps=5):
    """Local search: move one circle at a time to maximize LP sum of radii."""
    curr_sum = np.sum(radii)
    for _ in range(steps):
        improved = False
        order = np.random.permutation(N)
        for i in order:
            best_pos = centers[i].copy()
            best_sum = curr_sum
            
            # Try random local moves
            for _ in range(25):
                dx = np.random.uniform(-0.012, 0.012)
                dy = np.random.uniform(-0.012, 0.012)
                new_centers = centers.copy()
                new_centers[i] = np.clip(centers[i] + [dx, dy], 1e-5, 1.0 - 1e-5)
                
                r_new = solve_lp_radii(new_centers)
                s = np.sum(r_new)
                if s > best_sum:
                    best_sum = s
                    best_pos = new_centers[i].copy()
                    improved = True
                    
            if improved:
                centers[i] = best_pos
                radii = solve_lp_radii(centers)
                curr_sum = best_sum
        if not improved:
            break
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    for r0 in [0.085, 0.09, 0.095, 0.10, 0.105, 0.11]:
        for ang in np.linspace(-0.35, 0.35, 11):
            inits.append(make_hex_init(r0, ang))
    for s in range(10):
        inits.append(make_force_init(s))

    # Phase 2: Initial Optimization with LP refinement
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = solve_lp_radii(pts) * 0.995
        x0 = project_x(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                s_val = np.sum(r_new)
                if s_val > best_sum:
                    best_sum = s_val
                    best_x = res.x.copy()
                    best_x[2::3] = r_new
        except Exception:
            pass

    # Phase 3: Aggressive Topology Search with Deflation & Perturbation
    if best_x is not None:
        centers = np.column_stack((best_x[0::3], best_x[1::3]))
        radii = best_x[2::3]
        
        for cycle in range(80):
            x0 = best_x.copy()
            
            # Gradually recover radii to allow smooth topology changes
            shrink = 0.70 + 0.30 * (cycle / 80.0)
            x0[2::3] *= shrink
            
            # Decaying noise perturbation
            noise_scale = 0.004 / (cycle + 1)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            x0 = project_x(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    
                    x_p = res.x.copy()
                    x_p[2::3] = r_new
                    x_p = project_x(x_p)
                    
                    s_val = np.sum(x_p[2::3])
                    c_vals = constraints(x_p)
                    if np.min(c_vals) >= -1e-5 and s_val > best_sum:
                        best_sum = s_val
                        best_x = x_p.copy()
                        centers = c_tmp
                        radii = r_new
            except Exception:
                pass
                
        # Phase 4: Coordinate Descent Local Search
        for _ in range(5):
            centers, radii = coordinate_descent(centers, radii, steps=6)
            curr_sum = np.sum(radii)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_x = np.zeros(3 * N)
                best_x[0::3] = centers[:, 0]
                best_x[1::3] = centers[:, 1]
                best_x[2::3] = radii
                
        # Phase 5: High-precision polish on best solution
        x_pol = best_x.copy()
        x_pol[2::3] *= 0.99
        x_pol = project_x(x_pol)
        try:
            res = minimize(objective, x_pol, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                r_pol = solve_lp_radii(c_pol)
                x_pol = res.x.copy()
                x_pol[2::3] = r_pol
                x_pol = project_x(x_pol)
                s_pol = np.sum(x_pol[2::3])
                if np.min(constraints(x_pol)) >= -1e-5 and s_pol > best_sum:
                    best_sum = s_pol
                    best_x = x_pol.copy()
        except Exception:
            pass

    # Fallback
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        best_sum = np.sum(best_x[2::3])

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final LP squeeze to maximize radii for best centers
    final_r = solve_lp_radii(centers)
    if np.sum(final_r) > np.sum(radii) - 1e-7:
        radii = final_r.copy()

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
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])

    return centers, radii, float(np.sum(radii))
