# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000085 (state a6c5576f) state=5c33a9b3 sum of radii=2.625756 correctness=1.0
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
    """Returns all inequality constraints g(x) >= 0 (vectorized)."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints: 4*N
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
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
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, d)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.01)

def generate_hex_init(r0, angle_deg, seed):
    """Generates a rotated hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    y = 0.0
    row = 0
    while len(pts) < N + 20:
        x = (row % 2) * r0
        while x <= 1.0 + r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N+20])
    
    if angle_deg != 0:
        ang = np.deg2rad(angle_deg)
        c, s = np.cos(ang), np.sin(ang)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:,0]>=0.05) & (pts[:,0]<=0.95) & (pts[:,1]>=0.05) & (pts[:,1]<=0.95)
    pts = pts[mask]
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (pad, 2))])
    pts = pts[:N]
    pts += np.random.uniform(-0.001, 0.001, pts.shape)
    return pts

def force_init(seed):
    """Force-directed relaxation to spread points and push to boundaries."""
    np.random.seed(seed)
    centers = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(500):
        f = np.zeros_like(centers)
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[j] - centers[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-6:
                    rep = 0.01 / (d**2 + 0.001)
                    f[i] -= rep * dx
                    f[j] += rep * dx
            for dim in range(2):
                if centers[i, dim] < 0.15: f[i, dim] += 0.03
                elif centers[i, dim] > 0.85: f[i, dim] -= 0.03
        centers += f * 0.05
        centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initializations
    configs = []
    for r0 in [0.085, 0.09, 0.095, 0.10, 0.105]:
        for ang in [0, 5, 10, 15, 20, 25, 30, 45]:
            configs.append(generate_hex_init(r0, ang, seed=ang + int(r0*1000)))
    for s in range(12):
        configs.append(force_init(s))
        
    for c_init in configs:
        r_init = solve_lp_radii(c_init)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        # Project to strict bounds
        for i in range(N):
            r = max(1e-6, x0[3*i+2])
            x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
            x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0 - r)
            x0[3*i+2] = r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                vals = constraints(res.x)
                if np.min(vals) >= -1e-6 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Deflation-Perturbation Refinement
    if best_x is not None:
        for step in range(30):
            x0 = best_x.copy()
            x0[2::3] *= 0.985
            noise = 0.003 * (0.85 ** step)
            x0 += np.random.normal(0, noise, 3 * N)
            
            for i in range(N):
                r = max(1e-6, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0 - r)
                x0[3*i+2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    vals = constraints(res.x)
                    if np.min(vals) >= -1e-6 and curr > best_sum:
                        best_x = res.x.copy()
                        best_sum = curr
            except Exception:
                pass
                
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x[2::3]
    
    # Final strict validity adjustment
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i,0] < radii[i]-1e-11 or centers[i,0] > 1.0-radii[i]+1e-11 or \
               centers[i,1] < radii[i]-1e-11 or centers[i,1] > 1.0-radii[i]+1e-11:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    if np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i]+radii[j]-1e-11:
                        valid = False; break
                if not valid: break
        if valid: break
        
        radii *= 0.9995
        for i in range(N):
            centers[i,0] = np.clip(centers[i,0], radii[i], 1.0-radii[i])
            centers[i,1] = np.clip(centers[i,1], radii[i], 1.0-radii[i])
            
    return centers, radii, float(np.sum(radii))
