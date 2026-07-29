# sol_000162 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000147 (state 3da02ee3) state=aff8b4c3 sum of radii=2.627905 correctness=1.0
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
            
    bounds_lp = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-7)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_hex_init(r0, angle, scale=1.0):
    """Generates a hexagonal lattice initialization with optional rotation and scaling."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0 and len(pts) < N + 10:
            pts.append([x * scale, y * scale])
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
        pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (1, 2))])
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
    bounds_obj = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations + SLSQP
    inits = []
    for scale in [0.95, 1.0, 1.05, 1.1]:
        r0 = 0.09 * scale
        for ang in np.linspace(-0.35, 0.35, 9):
            inits.append(make_hex_init(r0, ang))
            
    for s in range(8):
        np.random.seed(s + 100)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        inits.append(pts)

    for c_init in inits:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        r_lp = solve_lp_radii(c_init)
        x0[2::3] = r_lp
        x0 = project_to_feasible(x0)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                x0 = res.x.copy()
                centers_p = np.column_stack((x0[0::3], x0[1::3]))
                r_p = solve_lp_radii(centers_p)
                if r_p is not None:
                    x0[2::3] = r_p
                    curr_sum = np.sum(r_p)
                    if curr_sum > best_sum and np.min(constraints(x0)) >= -1e-6:
                        best_sum = curr_sum
                        best_x = x0.copy()
        except Exception:
            pass

    # Phase 2: Simulated Annealing on Centers to escape local minima
    if best_x is not None:
        curr_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        best_val = np.sum(solve_lp_radii(curr_centers))
        best_centers_sa = curr_centers.copy()
        
        temp = 0.025
        for i in range(800):
            temp *= 0.9985
            k = np.random.randint(2, 6)
            idx = np.random.choice(N, k, replace=False)
            new_centers = curr_centers.copy()
            new_centers[idx] += np.random.normal(0, temp, (k, 2))
            new_centers = np.clip(new_centers, 1e-4, 1.0 - 1e-4)
            
            new_val = np.sum(solve_lp_radii(new_centers))
            delta = new_val - best_val
            
            if delta > 0 or (temp > 1e-9 and np.random.rand() < np.exp(delta / (temp + 1e-10))):
                curr_centers = new_centers
                if new_val > best_val:
                    best_val = new_val
                    best_centers_sa = new_centers.copy()
                    
        if best_val > best_sum:
            best_sum = best_val
            x_sa = np.zeros(3 * N)
            x_sa[0::3] = best_centers_sa[:, 0]
            x_sa[1::3] = best_centers_sa[:, 1]
            x_sa[2::3] = solve_lp_radii(best_centers_sa)
            best_x = x_sa.copy()

    # Phase 3: Greedy Coordinate Descent Fine-tuning
    if best_x is not None:
        curr_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        improved = True
        while improved:
            improved = False
            for i in range(N):
                best_local_sum = np.sum(best_x[2::3])
                for dx in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                        if dx == 0 and dy == 0: continue
                        new_centers = curr_centers.copy()
                        new_centers[i, 0] = np.clip(curr_centers[i, 0] + dx, 0.01, 0.99)
                        new_centers[i, 1] = np.clip(curr_centers[i, 1] + dy, 0.01, 0.99)
                        r_new = solve_lp_radii(new_centers)
                        s_new = np.sum(r_new)
                        if s_new > best_local_sum + 1e-7:
                            curr_centers = new_centers
                            best_local_sum = s_new
                            best_x[0::3] = curr_centers[:, 0]
                            best_x[1::3] = curr_centers[:, 1]
                            best_x[2::3] = r_new
                            best_sum = s_new
                            improved = True
                            break
                    if improved: break

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
