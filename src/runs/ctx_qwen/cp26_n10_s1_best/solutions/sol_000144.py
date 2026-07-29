# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000139 (state 7da59266) state=309e8fd9 sum of radii=2.628338 correctness=1.0
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
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist >= r_i + r_j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    # hypot is numerically stable and provides good gradients near contact
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
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
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_patterned_init(rows, angle=0.0, r0=0.09):
    """Generates a row-patterned hexagonal initialization with optional rotation."""
    pts = []
    y = r0
    row_idx = 0
    for count in rows:
        x_start = r0 if row_idx % 2 == 0 else 1.5 * r0
        for _ in range(count):
            if len(pts) >= N: break
            pts.append([x_start, y])
            x_start += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row_idx += 1
        
    # Pad if necessary
    while len(pts) < N:
        pts.append([np.random.uniform(0.15, 0.85), np.random.uniform(0.15, 0.85)])
        
    pts = np.array(pts[:N])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    return np.clip(pts, 0.02, 0.98)

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for step in range(200):
        f = np.zeros_like(pts)
        lr = 0.02 * (1.0 - step / 200.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.35 and d > 1e-5:
                    rep = 0.01 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.05
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.05
        pts += f * lr
        pts = np.clip(pts, 0.03, 0.97)
    return pts

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
    bounds_obj = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Generate Diverse Initial Configurations
    inits = []
    
    # Patterned hexagonal rows (various densities and rotations)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,6,7], [4,6,6,6,4], 
                [5,5,6,6,4], [6,4,6,5,5], [8,6,6,6], [5,7,5,7], [4,5,6,5,6], [6,7,6,7]]
    for p in patterns:
        for ang in np.linspace(-0.25, 0.25, 7):
            inits.append(make_patterned_init(p, ang))
            
    # Force-directed random layouts
    for s in range(12):
        inits.append(make_force_init(s))
        
    # Phase 2: Multi-start Optimization with LP-SLSQP Alternation
    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        
        # Initialize radii via LP for maximum head-start
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = np.maximum(r_lp * 0.99, 0.005)
        x0 = project_to_feasible(x0)
        
        # Primary SLSQP run
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                x0 = res.x.copy()
                # LP refinement on optimized centers
                c_tmp = np.column_stack((x0[0::3], x0[1::3]))
                r_new = solve_lp_radii(c_tmp)
                if r_new is not None:
                    x0[2::3] = r_new
                    x0 = project_to_feasible(x0)
                    
                curr_sum = np.sum(x0[2::3])
                if curr_sum > best_sum and np.min(constraints(x0)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = x0.copy()
        except Exception:
            pass
            
    # Phase 3: Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cyc in range(70):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise = 0.0025 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            
            # Deflate radii to create slack for repositioning
            x0[2::3] *= 0.95
            
            # Aggressively deflate a random subset to break rigid contact networks
            subset = np.random.choice(N, size=max(4, N // 4), replace=False)
            x0[subset * 3 + 2] *= 0.75
            
            x0 = project_to_feasible(x0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new = solve_lp_radii(c_tmp)
                    if r_new is not None:
                        x_p = res.x.copy()
                        x_p[2::3] = r_new * 0.999
                        x_p = project_to_feasible(x_p)
                        
                        s_val = np.sum(x_p[2::3])
                        if s_val > best_sum and np.min(constraints(x_p)) >= -1e-6:
                            best_sum = s_val
                            best_x = x_p.copy()
            except Exception:
                pass

        # Phase 4: High-Precision Polish with trust-constr
        x_pol = best_x.copy()
        x_pol[2::3] *= 0.99
        x_pol = project_to_feasible(x_pol)
        
        # Convert inequality constraints to format for trust-constr
        cons_trust = {'type': 'ineq', 'fun': constraints, 'jac': None}
        try:
            res_tc = minimize(objective, x_pol, method='trust-constr', bounds=bounds_obj,
                              constraints=cons_trust, options={'maxiter': 10000, 'verbose': 0})
            if not np.isnan(res_tc.fun):
                c_tc = np.column_stack((res_tc.x[0::3], res_tc.x[1::3]))
                r_tc = solve_lp_radii(c_tc)
                if r_tc is not None:
                    x_tc = res_tc.x.copy()
                    x_tc[2::3] = r_tc
                    x_tc = project_to_feasible(x_tc)
                    s_tc = np.sum(x_tc[2::3])
                    if s_tc > best_sum and np.min(constraints(x_tc)) >= -1e-6:
                        best_sum = s_tc
                        best_x = x_tc.copy()
        except Exception:
            pass

    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06
        best_sum = np.sum(best_x[2::3])
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
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
            
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
