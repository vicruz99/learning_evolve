# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000082 (state 67db05ee) state=517a2844 sum of radii=2.624724 correctness=1.0
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
    """Vectorized inequality constraints: g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
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
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, d)
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def make_hex_init(r0, angle):
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
        if y > 1.0 + r0:
            break
            
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(600):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-5:
                    rep = 0.015 / (d**2 + 0.001)
                    f[i] -= rep * dx
                    f[j] += rep * dx
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.04
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.04
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def project_to_feasible(x0):
    """Ensure optimization vector strictly respects bounds."""
    x0 = x0.copy()
    for i in range(N):
        r = max(1e-6, x0[3 * i + 2])
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
    configs = []
    # Hex grids with various rotations
    for r0 in [0.08, 0.09, 0.10]:
        for ang in np.linspace(-0.4, 0.4, 9):
            configs.append(make_hex_init(r0, ang))
    # Force directed layouts
    for s in range(10):
        configs.append(force_init(s))
        
    for c_init in configs:
        r_init = solve_lp_radii(c_init)
        if r_init is None:
            r_init = np.full(N, 0.06)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        x0 = project_to_feasible(x0)
        
        # Break symmetry slightly
        x0 += np.random.normal(0, 1e-5, x0.shape)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    curr_sum = np.sum(r_lp)
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_opt[:, 0]
                        best_x[1::3] = c_opt[:, 1]
                        best_x[2::3] = r_lp
        except Exception:
            continue
            
    # Phase 2: Deflation & Perturbation to escape local minima
    if best_x is not None:
        for step in range(30):
            x_pert = best_x.copy()
            # Shrink radii to create slack for repositioning
            x_pert[2::3] *= 0.95
            noise = 0.003 / (step + 1)
            x_pert[0::3] += np.random.normal(0, noise, N)
            x_pert[1::3] += np.random.normal(0, noise, N)
            
            x_pert = project_to_feasible(x_pert)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_lp = solve_lp_radii(c_opt)
                    if r_lp is not None:
                        curr_sum = np.sum(r_lp)
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_x = np.zeros(3 * N)
                            best_x[0::3] = c_opt[:, 0]
                            best_x[1::3] = c_opt[:, 1]
                            best_x[2::3] = r_lp
            except Exception:
                pass
                
    # Fallback (should not be reached)
    if best_x is None:
        c_fallback = make_hex_init(0.09, 0.0)
        r_fallback = solve_lp_radii(c_fallback)
        if r_fallback is None: r_fallback = np.full(N, 0.06)
        best_x = np.zeros(3 * N)
        best_x[0::3] = c_fallback[:, 0]
        best_x[1::3] = c_fallback[:, 1]
        best_x[2::3] = r_fallback
        best_sum = np.sum(r_fallback)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i,0] < radii[i]-1e-10 or centers[i,0] > 1.0-radii[i]+1e-10 or \
               centers[i,1] < radii[i]-1e-10 or centers[i,1] > 1.0-radii[i]+1e-10:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i]+radii[j]-1e-10:
                        valid = False; break
                if not valid: break
        if valid: break
        radii *= 0.9995
        centers[:,0] = np.clip(centers[:,0], radii, 1.0-radii)
        centers[:,1] = np.clip(centers[:,1], radii, 1.0-radii)
        
    return centers, radii, float(np.sum(radii))
