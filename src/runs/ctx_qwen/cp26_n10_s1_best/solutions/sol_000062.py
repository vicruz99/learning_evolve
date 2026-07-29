# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000052 (state 0d4d18bd) state=c526b615 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_IDX = np.triu_indices(N_CIRCLES, 1)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES

def constraints(x):
    """Vectorized inequality constraints g(x) >= 0 using squared distances."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist^2 >= (ri + rj)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    c = np.concatenate([
        c,
        dx[TRIU_IDX]**2 + dy[TRIU_IDX]**2 - dr[TRIU_IDX]**2
    ])
    return c

def objective(x):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N_CIRCLES
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x_val, y_val = centers[i]
        bound = min(x_val, 1.0 - x_val, y_val, 1.0 - y_val)
        A[idx, i] = 1.0
        b[idx] = bound
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = dist
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            # Slight shrinkage to guarantee strict feasibility for next steps
            return res.x * 0.9999
    except Exception:
        pass
    return np.full(n, 0.06)

def force_spread(centers, radii, steps=150):
    """Force-directed relaxation to push centers apart and maximize clearance."""
    pts = centers.copy()
    for _ in range(steps):
        f = np.zeros_like(pts)
        for i in range(N_CIRCLES):
            # Pairwise repulsion
            for j in range(i + 1, N_CIRCLES):
                dx = pts[j, 0] - pts[i, 0]
                dy = pts[j, 1] - pts[i, 1]
                d = np.hypot(dx, dy)
                if d < 0.35 and d > 1e-6:
                    rep = 0.015 / (d**2 + 0.001)
                    fx = rep * dx / d
                    fy = rep * dy / d
                    f[i] -= [fx, fy]
                    f[j] += [fx, fy]
            # Boundary repulsion
            for dim in range(2):
                if pts[i, dim] < radii[i] + 0.01:
                    f[i, dim] += 0.04
                if pts[i, dim] > 1.0 - radii[i] - 0.01:
                    f[i, dim] -= 0.04
                    
        pts += f * 0.05
        pts = np.clip(pts, 0.001, 0.999)
    return pts

def make_init_hex(seed, angle=0.0):
    """Generate a hexagonal lattice initialization with optional rotation."""
    np.random.seed(seed)
    pts = []
    r = 0.09
    y = r
    row = 0
    while len(pts) < N_CIRCLES + 5:
        x = r if row % 2 == 0 else 2.0 * r
        while x <= 1.0 - r:
            pts.append([x, y])
            x += 2.0 * r
        y += np.sqrt(3.0) * r
        row += 1
        if y > 1.0 + r:
            break
            
    pts = np.array(pts[:N_CIRCLES + 5])
    
    # Rotate around center
    if angle != 0.0:
        cx, cy = 0.5, 0.5
        ca, sa = np.cos(angle), np.sin(angle)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        pts[:, 0] = dx * ca - dy * sa + cx
        pts[:, 1] = dx * sa + dy * ca + cy
        
    pts = pts[:N_CIRCLES]
    pts += np.random.uniform(-0.005, 0.005, pts.shape)
    return np.clip(pts, 0.05, 0.95), np.full(N_CIRCLES, 0.085)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Generate diverse initial configurations
    inits = []
    # Rotated hexagonal lattices
    for s in range(15):
        c, r = make_init_hex(s, angle=s * 0.08)
        inits.append((c, r))
    # Random starts
    for _ in range(10):
        c = np.random.uniform(0.15, 0.85, (N_CIRCLES, 2))
        r = np.full(N_CIRCLES, 0.06)
        inits.append((c, r))
    # Grid start
    c_grid = np.zeros((N_CIRCLES, 2))
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N_CIRCLES:
                c_grid[idx] = [0.1 + 0.18 * j, 0.1 + 0.16 * i]
                idx += 1
    inits.append((c_grid, np.full(N_CIRCLES, 0.09)))
    
    # Primary optimization loop
    for c0, r0 in inits:
        c_curr, r_curr = c0.copy(), r0.copy()
        
        # Hybrid refinement: alternate LP radii and force-spreading centers
        for _ in range(3):
            r_curr = solve_lp_radii(c_curr)
            c_curr = force_spread(c_curr, r_curr, steps=100)
            
        # Flatten to optimization vector
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = c_curr[:, 0]
        x0[1::3] = c_curr[:, 1]
        x0[2::3] = r_curr
        
        # Project to feasible bounds for SLSQP stability
        x0[0::3] = np.clip(x0[0::3], 0.005, 0.995)
        x0[1::3] = np.clip(x0[1::3], 0.005, 0.995)
        x0[2::3] = np.clip(x0[2::3], 0.01, 0.3)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            
            if not np.isnan(res.fun):
                vals = constraints(res.x)
                if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Secondary refinement: perturb best solution and re-optimize
    if best_x is not None:
        for step in range(6):
            noise_scale = 0.002 / (step + 1)
            x_pert = best_x + np.random.normal(0, noise_scale, best_x.shape)
            r_pert = np.maximum(x_pert[2::3], 0.005)
            x_pert[0::3] = np.clip(x_pert[0::3], r_pert, 1.0 - r_pert)
            x_pert[1::3] = np.clip(x_pert[1::3], r_pert, 1.0 - r_pert)
            x_pert[2::3] = r_pert
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                vals = constraints(res.x)
                if np.min(vals) >= -1e-7 and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N_CIRCLES)
        best_x[2::3] = 0.06
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N_CIRCLES]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N_CIRCLES]
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N_CIRCLES):
            if radii[i] < 0:
                valid = False; break
            if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > 1.0 + 1e-9:
                valid = False; break
            if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > 1.0 + 1e-9:
                valid = False; break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False; break
                if not valid: break
        if valid: break
        # Minimal shrinkage to recover validity
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
