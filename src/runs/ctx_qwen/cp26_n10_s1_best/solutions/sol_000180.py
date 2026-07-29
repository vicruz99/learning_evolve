# sol_000180 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000167 (state 1af0cc64) state=7f440abf sum of radii=2.627905 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Returns all inequality constraints >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-8, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)] * N

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    num_ineq = n + n * (n - 1) // 2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A[idx, i] = 1.0
        b[idx] = max(0.0, lim)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, d)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex(r0, angle):
    """Generates a rotated hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 5:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    pts = np.array(pts[:N + 5])
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, 2)])
    return np.clip(pts[:N], 0.05, 0.95)

def force_spread(centers):
    """Force-directed relaxation to spread points evenly."""
    pts = centers.copy()
    for step in range(150):
        f = np.zeros_like(pts)
        lr = 0.02 * (1.0 - step / 150.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.35 and d > 1e-5:
                    rep = 0.01 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.08: f[i, dim] += 0.05
                elif pts[i, dim] > 0.92: f[i, dim] -= 0.05
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-8, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations + SLSQP Polish
    inits = []
    for scale in [0.95, 1.0, 1.05, 1.1]:
        for ang in np.linspace(-0.35, 0.35, 9):
            inits.append(make_hex(0.09 * scale, ang))
            
    for s in range(10):
        np.random.seed(s * 37 + 10)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        inits.append(force_spread(pts))
        
    for c0 in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        r_lp = solve_lp_radii(c0)
        x0[2::3] = r_lp * 0.99
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                x_tmp = res.x.copy()
                c_tmp = np.column_stack((x_tmp[0::3], x_tmp[1::3]))
                r_lp = solve_lp_radii(c_tmp)
                x_tmp[2::3] = r_lp
                x_tmp = project_to_feasible(x_tmp)
                s_val = np.sum(r_lp)
                if s_val > best_sum and np.min(constraint_func(x_tmp)) >= -1e-5:
                    best_sum = s_val
                    best_x = x_tmp.copy()
        except Exception:
            pass
            
    # Phase 2: Adaptive Coordinate Descent on Centers
    if best_x is not None:
        c_curr = np.column_stack((best_x[0::3], best_x[1::3]))
        r_curr = solve_lp_radii(c_curr)
        
        for it in range(25):
            step = 0.008 * (0.92 ** it)
            old_sum = np.sum(r_curr)
            for i in range(N):
                base_s = np.sum(r_curr)
                best_pos = c_curr[i].copy()
                
                # Grid perturbations: 4 cardinal + 4 diagonal
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step*0.707, step*0.707), (-step*0.707, step*0.707),
                               (step*0.707, -step*0.707), (-step*0.707, -step*0.707)]:
                    nc = c_curr.copy()
                    nc[i] = np.clip(c_curr[i] + [dx, dy], 1e-5, 1.0 - 1e-5)
                    r_tmp = solve_lp_radii(nc)
                    s_tmp = np.sum(r_tmp)
                    if s_tmp > base_s + 1e-7:
                        base_s = s_tmp
                        best_pos = nc[i].copy()
                        
                c_curr[i] = best_pos
                r_curr = solve_lp_radii(c_curr)
                
            if np.sum(r_curr) <= old_sum + 1e-8:
                break
                
        best_sum = np.sum(r_curr)
        best_x = np.zeros(3 * N)
        best_x[0::3] = c_curr[:, 0]
        best_x[1::3] = c_curr[:, 1]
        best_x[2::3] = r_curr

    # Phase 3: Simulated Annealing on Centers to Escape Local Minima
    if best_x is not None:
        curr_c = np.column_stack((best_x[0::3], best_x[1::3]))
        best_c = curr_c.copy()
        best_v = best_sum
        temp = 0.015
        
        for step in range(2000):
            temp *= 0.997
            k = np.random.randint(2, 6)
            idx = np.random.choice(N, k, replace=False)
            new_c = curr_c.copy()
            new_c[idx] += np.random.normal(0, temp, (k, 2))
            new_c = np.clip(new_c, 1e-4, 1.0 - 1e-4)
            
            v_new = np.sum(solve_lp_radii(new_c))
            delta = v_new - best_v
            
            if delta > 0 or (temp > 1e-9 and np.random.rand() < np.exp(delta / (temp + 1e-10))):
                curr_c = new_c
                if v_new > best_v:
                    best_v = v_new
                    best_c = new_c.copy()
                    
        if best_v > best_sum:
            best_sum = best_v
            best_x = np.zeros(3 * N)
            best_x[0::3] = best_c[:, 0]
            best_x[1::3] = best_c[:, 1]
            best_x[2::3] = solve_lp_radii(best_c)
            
    # Final strict validation and minimal repair against 1e-12 tolerance
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
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
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
