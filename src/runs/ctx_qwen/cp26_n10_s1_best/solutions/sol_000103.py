# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000084 (state de9b3486) state=0389f94b sum of radii=2.630179 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def objective_jac(x):
    """Analytical Jacobian of the objective."""
    jac = np.zeros(3 * N)
    jac[2::3] = -1.0
    return jac

def constraints(x):
    """Returns all inequality constraints g(x) >= 0."""
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
    
    i_idx, j_idx = np.tril_indices(N, -1)
    dist_sq = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2
    r_sum_sq = dr[i_idx, j_idx]**2
    
    c = np.concatenate([c, dist_sq - r_sum_sq])
    return c

def constraints_jac(x):
    """Returns the exact Jacobian matrix of the constraints."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    num_cons = 4 * N + N * (N - 1) // 2
    jac = np.zeros((num_cons, 3 * N))
    
    # Boundary constraints Jacobian
    for i in range(N):
        # xs[i] - rs[i] >= 0
        jac[i, 3 * i] = 1.0
        jac[i, 3 * i + 2] = -1.0
        # 1 - xs[i] - rs[i] >= 0
        jac[N + i, 3 * i] = -1.0
        jac[N + i, 3 * i + 2] = -1.0
        # ys[i] - rs[i] >= 0
        jac[2 * N + i, 3 * i + 1] = 1.0
        jac[2 * N + i, 3 * i + 2] = -1.0
        # 1 - ys[i] - rs[i] >= 0
        jac[3 * N + i, 3 * i + 1] = -1.0
        jac[3 * N + i, 3 * i + 2] = -1.0
        
    # Overlap constraints Jacobian
    idx = 4 * N
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    i_idx, j_idx = np.tril_indices(N, -1)
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        row = idx + k
        
        dxi = dx[i, j]
        dyi = dy[i, j]
        dri = dr[i, j]
        
        # Gradient w.r.t x[i], y[i], r[i]
        jac[row, 3 * i] = 2.0 * dxi
        jac[row, 3 * i + 1] = 2.0 * dyi
        jac[row, 3 * i + 2] = -2.0 * dri
        
        # Gradient w.r.t x[j], y[j], r[j]
        jac[row, 3 * j] = -2.0 * dxi
        jac[row, 3 * j + 1] = -2.0 * dyi
        jac[row, 3 * j + 2] = -2.0 * dri
        
    return jac

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
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
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def make_hex_init(r0, angle):
    """Generates a rotated hexagonal lattice initialization."""
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
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_init(seed):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(500):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[j] - pts[i]
                d = np.linalg.norm(diff)
                if d < 0.25 and d > 1e-5:
                    ff = 0.01 / (d**2 + 0.001)
                    f[i] -= ff * diff
                    f[j] += ff * diff
            for dim in range(2):
                if pts[i, dim] < 0.1: f[i, dim] += 0.03
                elif pts[i, dim] > 0.9: f[i, dim] -= 0.03
        pts += f * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    cons = {
        'type': 'ineq',
        'fun': constraints,
        'jac': constraints_jac
    }
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations
    configs = []
    for r0 in np.linspace(0.085, 0.105, 5):
        for ang in np.linspace(-0.4, 0.4, 11):
            configs.append(make_hex_init(r0, ang))
            
    for s in range(12):
        configs.append(force_init(s))
        
    for c_init in configs:
        r_lp = solve_lp_radii(c_init)
        if r_lp is None:
            r_lp = np.full(N, 0.08)
            
        # Ensure strict initial feasibility
        r_lp = np.maximum(r_lp * 0.99, 0.01)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_lp
        
        for i in range(N):
            x0[3*i] = np.clip(x0[3*i], r_lp[i], 1.0 - r_lp[i])
            x0[3*i+1] = np.clip(x0[3*i+1], r_lp[i], 1.0 - r_lp[i])
            
        try:
            res = minimize(objective, x0, method='SLSQP', jac=objective_jac,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                curr = -res.fun
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6 and curr > best_sum:
                    best_sum = curr
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        return np.zeros((N, 2)), np.zeros(N), 0.0
        
    # Phase 2: Iterative Refinement with Deflation & LP snapping
    for step in range(50):
        noise = 0.002 * (0.93 ** step)
        x0 = best_x + np.random.normal(0, noise, 3 * N)
        x0[2::3] *= 0.98  # Deflate radii to allow rearrangement
        
        for i in range(N):
            r = max(0.005, x0[3*i+2])
            x0[3*i] = np.clip(x0[3*i], r, 1.0 - r)
            x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0 - r)
            x0[3*i+2] = r
            
        try:
            res = minimize(objective, x0, method='SLSQP', jac=objective_jac,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            
            if not np.isnan(res.fun):
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6:
                    # LP refinement to snap radii to theoretical max for these centers
                    c_cur = res.x.reshape(N, 3)[:, :2]
                    r_lp = solve_lp_radii(c_cur)
                    if r_lp is not None:
                        r_lp = np.maximum(r_lp * 0.999, 0.0)
                        x_opt = res.x.copy()
                        x_opt[2::3] = r_lp
                        curr_lp = np.sum(r_lp)
                        if curr_lp > best_sum:
                            best_sum = curr_lp
                            best_x = x_opt
                    elif -res.fun > best_sum:
                        best_sum = -res.fun
                        best_x = res.x.copy()
        except Exception:
            pass
            
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x[2::3]
    
    # Final strict validity check and minimal numerical repair
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i,0] < radii[i]-1e-9 or centers[i,0] > 1-radii[i]+1e-9 or \
               centers[i,1] < radii[i]-1e-9 or centers[i,1] > 1-radii[i]+1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    if np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i]+radii[j]-1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
