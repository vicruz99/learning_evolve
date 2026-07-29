# sol_000193 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000181 (state 315c1ecb) state=aaeb0296 sum of radii=2.635896 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N

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
    
    # np.hypot provides stable gradients near contact points
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
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
            
    bounds_lp = [(0.0, None)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-9)
    except Exception:
        pass
    return np.full(n, 0.05)

def project_to_feasible(x):
    """Project variables to strictly satisfy bounds."""
    x = x.copy()
    for i in range(N):
        r = max(1e-8, x[3 * i + 2])
        x[3 * i] = np.clip(x[3 * i], r, 1.0 - r)
        x[3 * i + 1] = np.clip(x[3 * i + 1], r, 1.0 - r)
        x[3 * i + 2] = r
    return x

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
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (1, 2))])
    return pts[:N]

def make_force_init(seed):
    """Force-directed layout to spread points evenly and push to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(200):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.hypot(dx[0], dx[1])
                if d < 0.25 and d > 1e-5:
                    rep = 0.012 / (d**2 + 0.001)
                    f[i] -= dx * rep / d
                    f[j] += dx * rep / d
            for dim in range(2):
                if pts[i, dim] < 0.12: f[i, dim] += 0.05
                elif pts[i, dim] > 0.88: f[i, dim] -= 0.05
        pts += f * 0.04
        pts = np.clip(pts, 0.03, 0.97)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    # Hexagonal lattices with varied scales and rotations
    for r0 in [0.085, 0.090, 0.095, 0.100]:
        for ang in np.linspace(-0.4, 0.4, 11):
            inits.append(make_hex_init(r0, ang))
            
    # Force-directed spreads
    for s in range(15):
        inits.append(make_force_init(s))
        
    # Random starts
    for s in range(10):
        np.random.seed(s + 500)
        inits.append(np.random.uniform(0.15, 0.85, (N, 2)))

    # Phase 2: Multi-start Optimization with LP Initialization
    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        
        # Initialize radii via LP for maximum head-start
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = np.maximum(r_lp * 0.995, 1e-6)
        x0 = project_to_feasible(x0)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                if r_new is not None:
                    curr_sum = np.sum(r_new)
                    if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-6:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            continue

    # Phase 3: Deflation & Perturbation to Escape Local Minima
    if best_x is not None:
        for cycle in range(80):
            x0 = best_x.copy()
            
            # Cooling perturbation
            noise_scale = 0.003 * (0.90 ** cycle)
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            
            # Deflate a random subset to break rigid contact networks
            k = np.random.randint(3, 10)
            subset = np.random.choice(N, size=k, replace=False)
            x0[subset * 3 + 2] *= 0.80 + 0.20 * (cycle / 80.0)
            
            x0 = project_to_feasible(x0)

            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                if not np.isnan(res.fun):
                    centers_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(centers_p)
                    if r_p is not None:
                        x_p = res.x.copy()
                        x_p[2::3] = np.maximum(r_p * 0.998, 1e-6)
                        x_p = project_to_feasible(x_p)
                        
                        curr_sum = np.sum(x_p[2::3])
                        if curr_sum > best_sum and np.min(constraints(x_p)) >= -1e-6:
                            best_sum = curr_sum
                            best_x = x_p.copy()
            except Exception:
                pass

        # Phase 4: Rotation Search
        best_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-15, 15, 13):
            ang = np.deg2rad(ang_deg)
            c, s = np.cos(ang), np.sin(ang)
            rot_centers = (best_centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            rot_centers = np.clip(rot_centers, 0.01, 0.99)
            
            r_rot = solve_lp_radii(rot_centers)
            if r_rot is not None:
                x_rot = np.zeros(3 * N)
                x_rot[0::3] = rot_centers[:, 0]
                x_rot[1::3] = rot_centers[:, 1]
                x_rot[2::3] = r_rot * 0.99
                x_rot = project_to_feasible(x_rot)
                
                try:
                    res = minimize(objective, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                                   options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun):
                        centers_rot = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_final = solve_lp_radii(centers_rot)
                        if r_final is not None:
                            x_rot[2::3] = r_final
                            x_rot = project_to_feasible(x_rot)
                            curr_sum = np.sum(x_rot[2::3])
                            if curr_sum > best_sum and np.min(constraints(x_rot)) >= -1e-6:
                                best_sum = curr_sum
                                best_x = x_rot.copy()
                except Exception:
                    pass

        # Phase 5: Coordinate Descent Fine-Tuning
        curr_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        improved = True
        while improved:
            improved = False
            for i in range(N):
                best_local_sum = np.sum(best_x[2::3])
                # Try displacements in a grid
                for dx in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                        if dx == 0 and dy == 0:
                            continue
                        new_centers = curr_centers.copy()
                        new_centers[i, 0] = np.clip(curr_centers[i, 0] + dx, 0.01, 0.99)
                        new_centers[i, 1] = np.clip(curr_centers[i, 1] + dy, 0.01, 0.99)
                        
                        r_new = solve_lp_radii(new_centers)
                        s_new = np.sum(r_new)
                        if s_new > best_local_sum + 1e-8:
                            curr_centers = new_centers
                            best_local_sum = s_new
                            best_x[0::3] = curr_centers[:, 0]
                            best_x[1::3] = curr_centers[:, 1]
                            best_x[2::3] = r_new
                            best_sum = s_new
                            improved = True
                            break
                    if improved:
                        break

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
