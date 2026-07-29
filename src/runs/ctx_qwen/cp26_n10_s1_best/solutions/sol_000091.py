# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000070 (state 16cb787f) state=96c6b78b sum of radii=2.623621 correctness=1.0
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
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Overlap constraints: N*(N-1)/2 using squared distances for smooth gradients
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
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
    k = 0
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[k, i] = 1.0
        b_ub[k] = max(0.0, lim)
        k += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = max(0.0, d)
            k += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def make_hex_init(angle_deg, seed):
    """Generate a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    r = 0.13
    y = 0.0
    row = 0
    while len(pts) < N + 20:
        x = (row % 2) * r
        while x <= 1.0 + r:
            pts.append([x, y])
            x += 2.0 * r
        y += np.sqrt(3.0) * r
        row += 1
        
    pts = np.array(pts[:N + 20])
    c = np.array([0.5, 0.5])
    pts -= c
    ang = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ rot.T + c
    
    mask = (pts[:, 0] >= 0.05) & (pts[:, 0] <= 0.95) & (pts[:, 1] >= 0.05) & (pts[:, 1] <= 0.95)
    pts = pts[mask]
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:N]
    pts += np.random.uniform(-0.005, 0.005, pts.shape)
    return pts

def run_packing():
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multi-start optimization with diverse configurations
    configs = []
    for ang in np.linspace(0, 45, 12):
        for s in range(3):
            configs.append(make_hex_init(ang, s))
    for s in range(8):
        configs.append(np.random.uniform(0.15, 0.85, (N, 2)))
        
    for pts in configs:
        rs = np.full(N, 0.07)
        for i in range(N):
            m = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            for j in range(N):
                if i != j:
                    d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                    if d < m: m = d
            rs[i] = m * 0.35
            
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = rs
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_lp = solve_lp_radii(c_opt)
            if r_lp is not None:
                s_lp = np.sum(r_lp)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_x = np.zeros(3 * N)
                    best_x[0::3] = c_opt[:, 0]
                    best_x[1::3] = c_opt[:, 1]
                    best_x[2::3] = r_lp
            else:
                if np.min(constraints(res.x)) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Deflation & Perturbation Refinement to escape local minima
    if best_x is not None:
        for step in range(25):
            x0 = best_x.copy()
            scale = 0.98 - 0.001 * step
            x0[2::3] *= scale
            noise = 0.0025 * (0.92 ** step)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            for i in range(N):
                r = max(0.005, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 30000, 'ftol': 1e-13, 'disp': False})
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    s_lp = np.sum(r_lp)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_opt[:, 0]
                        best_x[1::3] = c_opt[:, 1]
                        best_x[2::3] = r_lp
                elif np.min(constraints(res.x)) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
                
    # Extract results
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Phase 3: Strict Validation & Minimal Numerical Repair
    for _ in range(100):
        ok = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or \
               centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9:
                ok = False
                break
        if ok:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            break
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
