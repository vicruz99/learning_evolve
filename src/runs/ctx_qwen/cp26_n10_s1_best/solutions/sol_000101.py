# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000089 (state 46a1566f) state=6c23d3d9 sum of radii=2.323782 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_IDX = np.triu_indices(N_CIRCLES, 1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Vectorized inequality constraints: g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    m = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    c = np.empty(m)
    
    # Boundary constraints
    c[0:N_CIRCLES] = cx - cr
    c[N_CIRCLES:2*N_CIRCLES] = 1.0 - cx - cr
    c[2*N_CIRCLES:3*N_CIRCLES] = cy - cr
    c[3*N_CIRCLES:4*N_CIRCLES] = 1.0 - cy - cr
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    idx = TRIU_IDX
    c[4*N_CIRCLES:] = dx[idx]**2 + dy[idx]**2 - dr[idx]**2
    return c

def constraint_jac(x):
    """Analytical Jacobian of the constraint function."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    m = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    n_vars = 3 * N_CIRCLES
    jac = np.zeros((m, n_vars))
    
    # Boundary constraints Jacobian
    jac[0:N_CIRCLES, 0::3] = 1.0
    jac[0:N_CIRCLES, 2::3] = -1.0
    
    jac[N_CIRCLES:2*N_CIRCLES, 0::3] = -1.0
    jac[N_CIRCLES:2*N_CIRCLES, 2::3] = -1.0
    
    jac[2*N_CIRCLES:3*N_CIRCLES, 1::3] = 1.0
    jac[2*N_CIRCLES:3*N_CIRCLES, 2::3] = -1.0
    
    jac[3*N_CIRCLES:4*N_CIRCLES, 1::3] = -1.0
    jac[3*N_CIRCLES:4*N_CIRCLES, 2::3] = -1.0
    
    # Overlap constraints Jacobian
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    i, j = TRIU_IDX
    k = np.arange(len(i))
    base = 4 * N_CIRCLES
    
    jac[base + k, 3 * i] = 2.0 * dx[i, j]
    jac[base + k, 3 * j] = -2.0 * dx[i, j]
    jac[base + k, 3 * i + 1] = 2.0 * dy[i, j]
    jac[base + k, 3 * j + 1] = -2.0 * dy[i, j]
    jac[base + k, 3 * i + 2] = -2.0 * dr[i, j]
    jac[base + k, 3 * j + 2] = -2.0 * dr[i, j]
    
    return jac

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N_CIRCLES
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
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(seed, angle, scale, shift_x=0.0, shift_y=0.0):
    """Generate a rotated, scaled, and shifted hexagonal lattice initialization."""
    np.random.seed(seed)
    pts = []
    y = 0.0
    row = 0
    r = 0.09 * scale
    while len(pts) < N_CIRCLES + 15:
        x = (row % 2) * r
        while x <= 1.0 + r:
            pts.append([x, y])
            x += 2.0 * r
        y += np.sqrt(3.0) * r
        row += 1
    pts = np.array(pts[:N_CIRCLES + 15])
    
    # Rotate around center
    center = np.array([0.5, 0.5])
    pts -= center
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = pts @ rot.T + center
    pts += [shift_x, shift_y]
    
    # Filter and pad
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    if len(pts) < N_CIRCLES:
        pad = N_CIRCLES - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    pts = pts[:N_CIRCLES]
    pts += np.random.uniform(-0.001, 0.001, pts.shape)
    return pts

def make_force_init(seed):
    """Generate initial configuration using repulsive force simulation pushing to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N_CIRCLES, 2))
    for step in range(1500):
        f = np.zeros_like(pts)
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-6:
                    ff = 0.015 / (d**2 + 0.001)
                    f[i] -= ff * dx
                    f[j] += ff * dx
            # Strong boundary attraction
            for dim in range(2):
                if pts[i, dim] < 0.08: f[i, dim] += 0.15
                elif pts[i, dim] > 0.92: f[i, dim] -= 0.15
        lr = 0.05 * (1.0 - step / 1500) ** 0.5
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_jac}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse multi-start optimization
    inits = []
    # Hexagonal grids with various rotations, scales, and shifts
    for seed in range(12):
        for ang in np.linspace(-0.2, 0.2, 5):
            for sc in [0.9, 1.0, 1.1]:
                pts = make_hex_init(seed, ang, sc, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05))
                r0 = solve_lp_radii(pts)
                x0 = np.zeros(3 * N_CIRCLES)
                x0[0::3] = pts[:, 0]
                x0[1::3] = pts[:, 1]
                x0[2::3] = r0
                inits.append(x0)
                
    # Force-directed starts
    for s in range(8):
        pts = make_force_init(s)
        r0 = solve_lp_radii(pts)
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r0
        inits.append(x0)
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraint_func(res.x)
                if np.min(c_vals) >= -1e-6 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating LP & SLSQP refinement with deflation
    if best_x is not None:
        for rnd in range(30):
            # LP step: maximize radii for current centers
            centers = np.column_stack((best_x[0::3], best_x[1::3]))
            r_lp = solve_lp_radii(centers)
            best_x[2::3] = r_lp
            
            # Deflate radii slightly to allow centers to move
            x0 = best_x.copy()
            x0[2::3] *= 0.980
            noise = 0.002 * (0.90 ** rnd)
            x0[0::3] += np.random.normal(0, noise, N_CIRCLES)
            x0[1::3] += np.random.normal(0, noise, N_CIRCLES)
            
            # Project back to strict bounds
            for i in range(N_CIRCLES):
                r = max(0.005, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c_vals = constraint_func(res.x)
                    if np.min(c_vals) >= -1e-6 and curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback
    if best_x is None:
        pts = make_hex_init(0, 0.0, 1.0)
        best_x = np.zeros(3 * N_CIRCLES)
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
        for i in range(N_CIRCLES):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-10 or centers[i, 0] > 1.0 - radii[i] + 1e-10 or \
               centers[i, 1] < radii[i] - 1e-10 or centers[i, 1] > 1.0 - radii[i] + 1e-10:
                valid = False
                break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
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
