# sol_000176 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000171 (state ed6c3974) state=789445b6 sum of radii=2.628596 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-8, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)])
    return b

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        xs - rs, 1.0 - xs - rs,
        ys - rs, 1.0 - ys - rs
    ])
    
    # Overlap constraints: dist >= r_i + r_j
    # np.hypot provides stable, non-zero gradients at contact points
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

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
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(angle, r0=0.092):
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
    
    c, s = np.cos(angle), np.sin(angle)
    pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return np.clip(pts[:N] + np.random.uniform(-0.002, 0.002, (N, 2)), 0.02, 0.98)

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for step in range(200):
        f = np.zeros_like(pts)
        lr = 0.012 * (1.0 - step / 200.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.linalg.norm(dx)
                if d < 0.25 and d > 1e-6:
                    rep = 0.005 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.04
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.04
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def project_x(x0):
    """Project variables to strictly satisfy bounds."""
    x0 = x0.copy()
    for i in range(N):
        r = max(1e-8, x0[3 * i + 2])
        x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
        x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
        x0[3 * i + 2] = r
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    # Rotated hexagonal lattices
    for ang in np.linspace(-0.4, 0.4, 17):
        inits.append(make_hex_init(ang))
    # Force-directed random spreads
    for s in range(12):
        inits.append(make_force_init(s))
    # Random starts
    for s in range(10):
        np.random.seed(s + 1000)
        inits.append(np.random.uniform(0.15, 0.85, (N, 2)))

    # Phase 2: Multi-start Optimization with LP Refinement
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        r_lp = solve_lp_radii(pts)
        x0[2::3] = r_lp * 0.97
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
        for cyc in range(100):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise_scale = 0.003 / (cyc + 1)
            subset_size = max(3, N // 4)
            subset = np.random.choice(N, size=subset_size, replace=False)
            
            x0[subset * 3] += np.random.normal(0, noise_scale, subset_size)
            x0[subset * 3 + 1] += np.random.normal(0, noise_scale, subset_size)
            x0[subset * 3 + 2] *= 0.65  # Deflate to break rigid contacts
            
            # Random position swap to break symmetries
            if cyc % 5 == 0:
                i, j = np.random.choice(N, 2, replace=False)
                x0[[i*3, j*3]] = x0[[j*3, i*3]]
                x0[[i*3+1, j*3+1]] = x0[[j*3+1, i*3+1]]
                
            x0 = project_x(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    
                    x_p = res.x.copy()
                    x_p[2::3] = r_new * 0.999
                    x_p = project_x(x_p)
                    
                    s_val = np.sum(x_p[2::3])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = x_p.copy()
            except Exception:
                pass

    # Phase 4: Boundary Pushing & High-Precision Polish
    if best_x is not None:
        # Try pushing centers slightly towards boundaries to see if LP radii increase
        c_curr = np.column_stack((best_x[0::3], best_x[1::3]))
        for _ in range(40):
            for i in range(N):
                bound = np.random.choice([0, 1, 2, 3])
                if bound == 0: c_curr[i, 0] -= 0.0015
                elif bound == 1: c_curr[i, 0] += 0.0015
                elif bound == 2: c_curr[i, 1] -= 0.0015
                else: c_curr[i, 1] += 0.0015
            c_curr = np.clip(c_curr, 0.01, 0.99)
            
            r_tmp = solve_lp_radii(c_curr)
            if np.sum(r_tmp) > best_sum:
                best_sum = np.sum(r_tmp)
                best_x = np.zeros(3 * N)
                best_x[0::3] = c_curr[:, 0]
                best_x[1::3] = c_curr[:, 1]
                best_x[2::3] = r_tmp
        
        # Final high-precision SLSQP polish
        for _ in range(5):
            x0 = best_x.copy()
            x0[2::3] *= 0.985
            x0 = project_x(x0)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
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

    # Fallback initialization (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        best_sum = np.sum(best_x[2::3])

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final LP squeeze to maximize radii for the best centers found
    final_r = solve_lp_radii(centers)
    if np.sum(final_r) > np.sum(radii) - 1e-9:
        radii = final_r.copy()

    # Final strict validation repair against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0: valid = False; break
            if centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9: valid = False; break
            if centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9: valid = False; break
                if not valid: break
        if valid: break
            
        radii *= 0.9995
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])

    return centers, radii, float(np.sum(radii))
