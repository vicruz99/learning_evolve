# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000184 (state 9759f81d) state=bacda5ba sum of radii=2.620921 correctness=1.0
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
    # np.hypot provides stable gradients at contact points compared to squared distance
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
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
    num_ineq = n + n * (n - 1) // 2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A[idx, i] = 1.0
        b[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, dist)
            idx += 1
            
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-6, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

def make_staggered_init():
    """Generates a staggered row initialization tailored for 26 circles."""
    pts = []
    # Pattern: 6, 5, 6, 5, 4 circles per row
    counts = [6, 5, 6, 5, 4]
    y = 0.10
    dy = 0.18
    for count in counts:
        dx = 0.80 / (count - 1) if count > 1 else 0.0
        x_start = 0.10 + (1.0 - 0.20 - (count - 1) * dx) / 2.0
        for k in range(count):
            pts.append([x_start + k * dx, y])
        y += dy
    return np.array(pts[:N])

def make_hex_init(angle, scale=1.0):
    """Generates a rotated hexagonal lattice initialization."""
    pts = []
    r0 = 0.095 * scale
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
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
    return pts[:N]

def run_packing():
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None

    # Phase 1: Diverse Initial Configurations
    inits = []
    # Staggered patterns with slight perturbations
    for _ in range(4):
        inits.append(make_staggered_init() + np.random.uniform(-0.004, 0.004, (N, 2)))
        
    # Rotated hexagonal lattices
    for ang in np.linspace(-0.25, 0.25, 15):
        inits.append(make_hex_init(ang, scale=1.0))
    for ang in np.linspace(0.3, 0.55, 12):
        inits.append(make_hex_init(ang, scale=0.92))

    # Phase 2: Multi-start SLSQP Optimization
    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        
        # Initialize radii via LP for maximum head-start
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = np.maximum(r_lp * 0.995, 1e-6)
        x0 = project_to_feasible(x0)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                curr_sum = np.sum(r_new)
                
                if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = res.x.copy()
                    best_x[2::3] = r_new
        except Exception:
            continue

    # Phase 3: Deflation & Repositioning to Escape Local Minima
    if best_x is not None:
        for cyc in range(50):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise_scale = 0.0025 * (0.91 ** cyc)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 8)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.80
            
            x0 = project_to_feasible(x0)

            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    centers_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(centers_p)
                    if r_p is not None:
                        curr_sum = np.sum(r_p)
                        if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-6:
                            best_sum = curr_sum
                            best_x = res.x.copy()
                            best_x[2::3] = r_p
            except Exception:
                pass

        # Phase 4: Coordinate Descent Fine-Tuning
        curr_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        curr_radii = best_x[2::3]
        best_local_sum = np.sum(curr_radii)
        
        for _ in range(5):
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    best_step_sum = best_local_sum
                    best_step_c = curr_centers[i].copy()
                    
                    # Try displacements in 8 directions
                    for dx in [-0.004, -0.001, 0.001, 0.004]:
                        for dy in [-0.004, -0.001, 0.001, 0.004]:
                            new_centers = curr_centers.copy()
                            new_centers[i, 0] = np.clip(curr_centers[i, 0] + dx, 0.005, 0.995)
                            new_centers[i, 1] = np.clip(curr_centers[i, 1] + dy, 0.005, 0.995)
                            
                            r_new = solve_lp_radii(new_centers)
                            s_new = np.sum(r_new)
                            if s_new > best_step_sum + 1e-8:
                                best_step_sum = s_new
                                best_step_c = new_centers[i].copy()
                                
                    if best_step_sum > best_local_sum + 1e-8:
                        curr_centers[i] = best_step_c
                        best_local_sum = best_step_sum
                        curr_radii = solve_lp_radii(curr_centers)
                        best_x[0::3] = curr_centers[:, 0]
                        best_x[1::3] = curr_centers[:, 1]
                        best_x[2::3] = curr_radii
                        best_sum = best_local_sum
                        improved = True

        # Phase 5: Final High-Precision Polish
        x0 = best_x.copy()
        x0[2::3] *= 0.998
        x0 = project_to_feasible(x0)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                c_f = np.column_stack((res.x[0::3], res.x[1::3]))
                r_f = solve_lp_radii(c_f)
                s_f = np.sum(r_f)
                if s_f > best_sum:
                    best_sum = s_f
                    best_x = res.x.copy()
                    best_x[2::3] = r_f
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

    # Final strict validity adjustment against 1e-12 tolerance
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
