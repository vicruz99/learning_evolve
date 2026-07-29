# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000070 (state 16cb787f) state=46a1566f sum of radii=2.634292 correctness=1.0
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

def constraints(x):
    """Compute all boundary and non-overlap constraints as a vector >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([cx - cr, 1.0 - cx - cr, cy - cr, 1.0 - cy - cr])
    
    # Overlap constraints: N*(N-1)/2
    # Vectorized distance and radius sum calculations
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

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
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.01)

def make_hex_init(seed, angle, scale):
    """Generate a rotated and scaled hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    y = 0.0
    row = 0
    r = 0.09 * scale
    while len(pts) < N + 10:
        x = (row % 2) * r
        while x <= 1.0 + r:
            pts.append([x, y])
            x += 2.0 * r
        y += np.sqrt(3.0) * r
        row += 1
    pts = np.array(pts[:N + 10])
    
    # Rotate around center
    center = np.array([0.5, 0.5])
    pts -= center
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = pts @ rot.T + center
    
    # Filter and pad
    mask = (pts[:, 0] >= 0.05) & (pts[:, 0] <= 0.95) & (pts[:, 1] >= 0.05) & (pts[:, 1] <= 0.95)
    pts = pts[mask]
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:N]
    pts += np.random.uniform(-0.001, 0.001, pts.shape)
    return pts

def make_force_init(seed):
    """Generate initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(300):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-6:
                    ff = 0.01 / (d**2 + 0.001)
                    f[i] -= ff * dx
                    f[j] += ff * dx
            if pts[i, 0] < 0.1: f[i, 0] += 0.02
            elif pts[i, 0] > 0.9: f[i, 0] -= 0.02
            if pts[i, 1] < 0.1: f[i, 1] += 0.02
            elif pts[i, 1] > 0.9: f[i, 1] -= 0.02
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse starts (Hexagonal + Force-directed)
    inits = []
    for seed in range(10):
        for ang in [-0.2, 0.0, 0.2]:
            for sc in [0.95, 1.05]:
                pts = make_hex_init(seed, ang, sc)
                radii = solve_lp_radii(pts)
                x0 = np.zeros(3 * N)
                x0[0::3] = pts[:, 0]
                x0[1::3] = pts[:, 1]
                x0[2::3] = radii
                inits.append(x0)
                
    for seed in range(5):
        pts = make_force_init(seed)
        radii = solve_lp_radii(pts)
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = radii
        inits.append(x0)
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating LP & SLSQP refinement
    if best_x is not None:
        for rnd in range(25):
            centers = np.column_stack((best_x[0::3], best_x[1::3]))
            r_lp = solve_lp_radii(centers)
            best_x[2::3] = r_lp
            
            # Deflate radii slightly to allow centers to move
            x0 = best_x.copy()
            x0[2::3] *= 0.985
            noise = 0.001 * (0.95 ** rnd)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Project back to strict bounds
            for i in range(N):
                r = max(0.005, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-6 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Phase 3: Aggressive random restart from best to escape local minima
    if best_x is not None:
        for rnd in range(10):
            x0 = best_x.copy()
            x0[2::3] *= 0.90
            x0[0::3] += np.random.normal(0, 0.005, N)
            x0[1::3] += np.random.normal(0, 0.005, N)
            for i in range(N):
                r = max(0.01, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-6 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
            except Exception:
                pass

    # Fallback (should not be reached)
    if best_x is None:
        pts = make_hex_init(0, 0.0, 1.0)
        best_x = np.zeros(3 * N)
        best_x[0::3] = pts[:, 0]
        best_x[1::3] = pts[:, 1]
        best_x[2::3] = 0.085
        best_sum = -objective(best_x)
        
    # Extract centers and radii
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final strict validation & minimal numerical repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-10 or centers[i, 0] > 1.0 - radii[i] + 1e-10 or \
               centers[i, 1] < radii[i] - 1e-10 or centers[i, 1] > 1.0 - radii[i] + 1e-10:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Gentle shrinkage to guarantee strict compliance
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
