# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000119 (state 4956bde4) state=6d234c93 sum of radii=2.626678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
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
    return np.full(n, 0.05)

def lp_objective(centers):
    """Objective function for center-only optimization: negative sum of LP radii."""
    r = solve_lp_radii(centers)
    return -np.sum(r)

def make_hex_init(r0, angle):
    """Generates a rotated hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 10:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    pts = np.array(pts[:N + 10])
    
    center = np.array([0.5, 0.5])
    pts -= center
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, -s], [s, c]])
    pts += center
    
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    if len(pts) < N:
        pad = N - len(pts)
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (pad, 2))])
    return pts[:N]

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    for r0 in [0.085, 0.09, 0.095, 0.10]:
        for ang in np.linspace(-0.4, 0.4, 13):
            pts = make_hex_init(r0, ang)
            inits.append(pts)
            
    for seed in range(8):
        np.random.seed(seed + 200)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Quick force spread
        for _ in range(100):
            f = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.3 and d > 1e-5:
                        rep = 0.01 / (d**2 + 0.001)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
            pts += f * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)

    # Phase 2: Multi-start Optimization (LP -> Center Opt -> Joint SLSQP)
    for pts in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        
        r_lp = solve_lp_radii(pts)
        x0[2::3] = np.maximum(r_lp * 0.995, 0.005)
        x0 = project_to_feasible(x0)
        
        # Optimize centers only using LP objective
        try:
            res_c = minimize(lp_objective, x0[0::3], method='Nelder-Mead', 
                             bounds=[(0.0, 1.0)]*N + [(0.0, 1.0)]*N,
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            x0[0::3] = res_c.x[:N]
            x0[1::3] = res_c.x[N:]
        except Exception:
            pass

        # Update radii via LP on optimized centers
        c_tmp = np.column_stack((x0[0::3], x0[1::3]))
        x0[2::3] = solve_lp_radii(c_tmp) * 0.998
        x0 = project_to_feasible(x0)
        
        # Joint refinement with SLSQP
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                s = -res.fun
                vals = constraints(res.x)
                if np.min(vals) >= -1e-5 and s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Basin Hopping with Advanced Perturbations
    if best_x is not None:
        for cycle in range(50):
            noise_scale = 0.003 * (0.93 ** cycle)
            x_p = best_x.copy()
            
            # Strategy 1: Global Rotation
            if cycle % 4 == 0:
                ang = np.random.uniform(-0.15, 0.15)
                c, s = np.cos(ang), np.sin(ang)
                rot_mat = np.array([[c, -s], [s, c]])
                centers = np.column_stack((x_p[0::3], x_p[1::3]))
                centers = (centers - 0.5) @ rot_mat + 0.5
                centers = np.clip(centers, 0.02, 0.98)
                x_p[0::3] = centers[:, 0]
                x_p[1::3] = centers[:, 1]
            
            # Strategy 2: Deflation of subset
            subset = np.random.choice(N, size=max(1, N//4), replace=False)
            x_p[subset * 3 + 2] *= 0.85
            
            # Strategy 3: Gaussian noise
            x_p[0::3] += np.random.normal(0, noise_scale, N)
            x_p[1::3] += np.random.normal(0, noise_scale, N)
            
            x_p = project_to_feasible(x_p)
            
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_opt)
                    if r_new is not None:
                        x_final = res.x.copy()
                        x_final[2::3] = r_new * 0.999
                        x_final = project_to_feasible(x_final)
                        
                        s = np.sum(x_final[2::3])
                        vals = constraints(x_final)
                        if np.min(vals) >= -1e-5 and s > best_sum:
                            best_sum = s
                            best_x = x_final.copy()
            except Exception:
                pass

    # Fallback
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        best_sum = np.sum(best_x[2::3])
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final strict validation repair against 1e-12 tolerance
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
            
        radii *= 0.9998
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
