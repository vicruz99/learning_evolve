# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000107 (state 898129cd) state=b6c9871b sum of radii=2.630179 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Vectorized inequality constraints: g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints: dist^2 >= (ri + rj)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
    i_idx, j_idx = np.tril_indices(N, -1)
    c = np.concatenate([c, dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2 - dr[i_idx, j_idx]**2])
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
        lim = min(centers[i, 0], 1.0 - centers[i, 0], 
                  centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(1e-9, lim)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(1e-9, d)
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.01)

def project_to_bounds(x):
    """Ensure optimization vector strictly respects bounds."""
    r = np.maximum(x[2::3], 1e-7)
    x[0::3] = np.clip(x[0::3], r, 1.0 - r)
    x[1::3] = np.clip(x[1::3], r, 1.0 - r)
    x[2::3] = r
    return x

def hex_init(scale, angle):
    """Generates a hexagonal lattice initialization."""
    pts = []
    r = 0.09 * scale
    y = r
    row = 0
    while len(pts) < N + 5:
        x = r if row % 2 == 0 else 2.0 * r
        while x <= 1.0 - r + 0.05:
            pts.append([x, y])
            x += 2.0 * r
        y += np.sqrt(3.0) * r
        row += 1
        if y > 1.0 + r:
            break
            
    pts = np.array(pts[:N + 5])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        cx, cy = 0.5, 0.5
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        pts[:, 0] = dx * c - dy * s + cx
        pts[:, 1] = dx * s + dy * c + cy
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & \
           (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (1, 2))])
    return pts[:N]

def force_init(seed, steps=500):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for step in range(steps):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-6:
                    rep = 0.015 / (d**2 + 0.001)
                    f[i] -= rep * dx
                    f[j] += rep * dx
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.04
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.04
        lr = 0.05 * (1.0 - step / steps)
        pts += f * lr
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initializations
    inits = []
    
    # Hexagonal lattices with fine-grained scale and rotation search
    for scale in np.linspace(0.90, 1.15, 6):
        for ang in np.linspace(-0.3, 0.3, 11):
            inits.append(hex_init(scale, ang))
            
    # Force-directed layouts
    for s in range(12):
        inits.append(force_init(s))
        
    # Multi-start optimization
    for c_init in inits:
        r_init = solve_lp_radii(c_init)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        x0 = project_to_bounds(x0)
        
        # Break exact symmetries
        x0 += np.random.normal(0, 1e-5, x0.shape)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    curr_sum = np.sum(r_lp)
                    c_vals = constraints(np.zeros(3*N)) # dummy check
                    # Reconstruct x for constraint check if needed, but LP guarantees feasibility
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_opt[:, 0]
                        best_x[1::3] = c_opt[:, 1]
                        best_x[2::3] = r_lp
        except Exception:
            continue
            
    # Phase 2: Deflation & Perturbation Refinement
    if best_x is not None:
        for step in range(30):
            x_pert = best_x.copy()
            # Shrink radii to create slack for repositioning
            x_pert[2::3] *= 0.96
            noise = 0.0025 / (step + 1)
            x_pert[0::3] += np.random.normal(0, noise, N)
            x_pert[1::3] += np.random.normal(0, noise, N)
            x_pert = project_to_bounds(x_pert)
            
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
                
    # Fallback
    if best_x is None:
        c_fb = hex_init(1.0, 0.0)
        r_fb = solve_lp_radii(c_fb)
        best_x = np.zeros(3 * N)
        best_x[0::3] = c_fb[:, 0]
        best_x[1::3] = c_fb[:, 1]
        best_x[2::3] = r_fb
        best_sum = np.sum(r_fb)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Final strict validity adjustment
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
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9998
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
