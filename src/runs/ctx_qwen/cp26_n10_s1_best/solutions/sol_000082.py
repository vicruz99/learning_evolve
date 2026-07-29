# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000061 (state 63a33892) state=67db05ee sum of radii=2.627681 correctness=1.0
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

def get_constraints(x):
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
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-6, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        b[idx] = max(0.0, lim)
        A[idx, i] = 1.0
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            b[idx] = max(0.0, d)
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(0.0, None)]*n, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def force_init(seed):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(800):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i+1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.3 and d > 1e-5:
                    rep = 0.02 / (d**2 + 0.001)
                    f[i] -= rep * dx
                    f[j] += rep * dx
            for dim in range(2):
                if pts[i, dim] < 0.15: f[i, dim] += 0.05
                elif pts[i, dim] > 0.85: f[i, dim] -= 0.05
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def hex_init(pat, r0, angle=0.0):
    """Generates a hexagonal lattice initialization."""
    centers = np.zeros((N, 2))
    idx = 0
    y = r0
    y_step = np.sqrt(3.0) * r0
    for i, cnt in enumerate(pat):
        x_start = r0 + (r0 if i%2==1 else 0.0)
        for k in range(cnt):
            if idx < N:
                centers[idx, 0] = x_start + k * 2.0 * r0
                centers[idx, 1] = y
                idx += 1
        y += y_step
    while idx < N:
        centers[idx, 0] = np.random.uniform(0.1, 0.9)
        centers[idx, 1] = np.random.uniform(0.1, 0.9)
        idx += 1
        
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        cx, cy = 0.5, 0.5
        dx = centers[:,0] - cx
        dy = centers[:,1] - cy
        centers[:,0] = dx*c - dy*s + cx
        centers[:,1] = dx*s + dy*c + cy
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initializations
    configs = []
    pats = [[5,6,5,6,4], [6,5,6,5,4], [5,6,6,5,4], [4,6,6,6,4], [5,5,5,5,6]]
    for p in pats:
        for a in [0.0, 0.15, -0.15, 0.3, -0.3, 0.5]:
            configs.append(hex_init(p, 0.085, a))
    for s in range(12):
        configs.append(force_init(s))
        
    for c_init in configs:
        r_init = solve_lp_radii(c_init)
        if r_init is None:
            r_init = np.full(N, 0.06)
            
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:,0]
        x0[1::3] = c_init[:,1]
        x0[2::3] = r_init
        
        # Break symmetry
        x0 += np.random.normal(0, 1e-5, x0.shape)
        
        # Ensure strict bounds for optimizer
        r_buf = np.maximum(x0[2::3], 1e-5)
        x0[0::3] = np.clip(x0[0::3], r_buf, 1.0-r_buf)
        x0[1::3] = np.clip(x0[1::3], r_buf, 1.0-r_buf)
        
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
                        best_x = np.zeros(3*N)
                        best_x[0::3] = c_opt[:,0]
                        best_x[1::3] = c_opt[:,1]
                        best_x[2::3] = r_lp
        except Exception:
            continue
            
    # Phase 2: Deflation & Refinement to escape local minima
    if best_x is not None:
        for step in range(25):
            x_pert = best_x.copy()
            x_pert[2::3] *= 0.96
            noise = 0.004 / (step + 1)
            x_pert[0::3] += np.random.normal(0, noise, N)
            x_pert[1::3] += np.random.normal(0, noise, N)
            
            r_p = np.maximum(x_pert[2::3], 1e-5)
            x_pert[0::3] = np.clip(x_pert[0::3], r_p, 1.0-r_p)
            x_pert[1::3] = np.clip(x_pert[1::3], r_p, 1.0-r_p)
            x_pert[2::3] = r_p
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_lp = solve_lp_radii(c_opt)
                    if r_lp is not None:
                        curr_sum = np.sum(r_lp)
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_x = np.zeros(3*N)
                            best_x[0::3] = c_opt[:,0]
                            best_x[1::3] = c_opt[:,1]
                            best_x[2::3] = r_lp
            except Exception:
                pass
                
    # Fallback
    if best_x is None:
        c_fallback = hex_init([5,6,5,6,4], 0.08, 0.0)
        r_fallback = solve_lp_radii(c_fallback)
        if r_fallback is None: r_fallback = np.full(N, 0.06)
        best_x = np.zeros(3*N)
        best_x[0::3] = c_fallback[:,0]
        best_x[1::3] = c_fallback[:,1]
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
