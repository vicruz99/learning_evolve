# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000118 (state 224f6ad6) state=fb45e5b9 sum of radii=2.625313 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, dx[TRIL]**2 + dy[TRIL]**2 - dr[TRIL]**2])
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
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.06)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def generate_init(strategy, seed=0):
    """Generate initial center configurations based on strategy."""
    np.random.seed(seed)
    pts = np.zeros((N, 2))
    
    if strategy == 'hex':
        r0 = 0.09
        idx = 0
        y = r0
        row = 0
        while idx < N:
            x = r0 if row % 2 == 0 else 2.0 * r0
            while x <= 1.0 - r0 and idx < N:
                pts[idx] = [x, y]
                idx += 1
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
    elif strategy == 'grid':
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < N:
                    pts[idx] = [0.1 + j * 0.18, 0.1 + i * 0.16]
                    idx += 1
    elif strategy == 'boundary':
        # Explicitly bias towards corners and edges where circles can be larger
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        edges = [[0.5, 0.06], [0.06, 0.5], [0.5, 0.94], [0.94, 0.5]]
        pts[:8] = corners + edges
        r0 = 0.08
        idx = 8
        y = r0
        row = 0
        while idx < N:
            x = r0 if row % 2 == 0 else 2.0 * r0
            while x <= 1.0 - r0 and idx < N:
                pts[idx] = [x, y]
                idx += 1
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
    elif strategy == 'rand':
        pts = np.random.uniform(0.1, 0.9, (N, 2))
    else: # force
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for step in range(300):
            f = np.zeros_like(pts)
            lr = 0.02 * (1.0 - step / 300)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.3 and d > 1e-5:
                        rep = 0.01 / (d**2 + 0.001)
                        f[i] -= dx * rep / d
                        f[j] += dx * rep / d
                for dim in range(2):
                    if pts[i, dim] < 0.15: f[i, dim] += 0.05
                    elif pts[i, dim] > 0.85: f[i, dim] -= 0.05
            pts += f * lr
            pts = np.clip(pts, 0.05, 0.95)
            
    # Slight perturbation to break exact symmetries
    pts += np.random.uniform(-0.003, 0.003, pts.shape)
    return np.clip(pts, 0.01, 0.99)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multi-start Optimization
    strategies = ['hex', 'grid', 'boundary', 'force']
    for strat in strategies:
        for seed in range(6):
            pts = generate_init(strat, seed)
            x0 = np.zeros(3 * N)
            x0[0::3] = pts[:, 0]
            x0[1::3] = pts[:, 1]
            
            # Initialize radii via LP for maximum head-start
            r_lp = solve_lp_radii(pts)
            x0[2::3] = np.maximum(r_lp * 0.995, 0.01)
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    s = -res.fun
                    vals = constraints(res.x)
                    if np.min(vals) >= -1e-5 and s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass

    # Phase 2: Iterative Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cycle in range(50):
            x0 = best_x.copy()
            
            # Deflate a random subset of radii to break rigid contacts
            subset = np.random.choice(N, size=N // 2, replace=False)
            x0[subset * 3 + 2] *= 0.88
            
            # Perturb centers with cooling noise
            noise = 0.0015 * (1.0 - cycle / 50.0)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        s = np.sum(r_new)
                        if s > best_sum:
                            best_sum = s
                            best_x = np.zeros(3 * N)
                            best_x[0::3] = c_tmp[:, 0]
                            best_x[1::3] = c_tmp[:, 1]
                            best_x[2::3] = r_new
                            
                            # Quick polish on the new best
                            x0_polish = best_x.copy()
                            x0_polish[2::3] *= 0.998
                            x0_polish = project_to_feasible(x0_polish)
                            try:
                                res2 = minimize(objective, x0_polish, method='SLSQP', bounds=bounds, 
                                                constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                                if not np.isnan(res2.fun):
                                    c_pol = np.column_stack((res2.x[0::3], res2.x[1::3]))
                                    r_pol = solve_lp_radii(c_pol)
                                    if r_pol is not None and np.sum(r_pol) > best_sum:
                                        best_sum = np.sum(r_pol)
                                        best_x[0::3] = c_pol[:, 0]
                                        best_x[1::3] = c_pol[:, 1]
                                        best_x[2::3] = r_pol
                            except Exception:
                                pass
            except Exception:
                pass

    # Fallback (should not be reached with valid inits)
    if best_x is None:
        pts = generate_init('hex', 0)
        best_x = np.zeros(3 * N)
        best_x[0::3] = pts[:, 0]
        best_x[1::3] = pts[:, 1]
        best_x[2::3] = solve_lp_radii(pts)
        best_sum = np.sum(best_x[2::3])
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final strict validation & minimal numerical repair
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
            
        # Gentle shrinkage to guarantee strict compliance with validator tolerance
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
