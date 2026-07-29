# sol_000196 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000181 (state 315c1ecb) state=b7b03371 sum of radii=2.614689 correctness=1.0
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

def constraints_sq(x):
    """Returns all inequality constraints >= 0 (vectorized, squared distances for smooth gradients)."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    num_ineq = n + n*(n-1)//2
    A = np.zeros((num_ineq, n))
    b = np.zeros(num_ineq)
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        A[idx, i] = 1.0
        b[idx] = max(0.0, lim)
        idx += 1
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, d)
            idx += 1
    bounds_lp = [(0.0, None)]*n
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds_lp, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.05)

def make_staggered_init():
    """Generates a 6-5-6-5-4 staggered grid initialization."""
    pts = []
    rows = [6, 5, 6, 5, 4]
    y_start = 0.06
    y_step = 0.18
    for r_idx, count in enumerate(rows):
        y = y_start + r_idx * y_step
        x_step = 0.165
        x_start = (1.0 - (count-1)*x_step) / 2.0
        if r_idx % 2 == 1:
            x_start += x_step / 2.0
        for k in range(count):
            x = x_start + k * x_step
            pts.append([x, y])
    return np.array(pts[:N])

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
    cons = {'type': 'ineq', 'fun': constraints_sq}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations
    inits = [make_staggered_init()]
    for ang in np.linspace(-0.35, 0.35, 11):
        inits.append(make_hex_init(0.092, ang))
    for s in range(12):
        np.random.seed(s + 100)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        inits.append(pts)

    # Phase 2: SLSQP on diverse starts
    for c_init in inits:
        r_lp = solve_lp_radii(c_init)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_lp * 0.995
        x0 = project_to_feasible(x0)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                c_tmp = np.column_stack((res.x[0::3], res.x[1::3]))
                r_new = solve_lp_radii(c_tmp)
                s_val = np.sum(r_new)
                if s_val > best_sum and np.min(constraints_sq(res.x)) >= -1e-5:
                    best_sum = s_val
                    best_x = res.x.copy()
                    best_x[2::3] = r_new
        except Exception:
            pass

    # Phase 3: Simulated Annealing on centers + Periodic SLSQP Polish
    if best_x is not None:
        curr_c = np.column_stack((best_x[0::3], best_x[1::3]))
        curr_r = best_x[2::3]
        best_sum = np.sum(curr_r)
        T = 0.06
        
        for step in range(1000):
            move_type = np.random.rand()
            new_c = curr_c.copy()
            
            if move_type < 0.65:
                # Local perturbation
                idx = np.random.choice(N, np.random.randint(2, 6), replace=False)
                new_c[idx] += np.random.normal(0, 0.004, (len(idx), 2))
            elif move_type < 0.85:
                # Global rotation/translation
                ang = np.random.uniform(-0.04, 0.04)
                c, s = np.cos(ang), np.sin(ang)
                new_c = (new_c - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
                new_c += np.random.uniform(-0.01, 0.01, (N, 2))
            else:
                # Deflation strategy to break rigid contact networks
                new_c += np.random.normal(0, 0.012, (N, 2))
                
            new_c = np.clip(new_c, 0.005, 0.995)
            new_r = solve_lp_radii(new_c)
            new_sum = np.sum(new_r)
            
            # Metropolis acceptance criterion
            if new_sum > best_sum or (T > 1e-6 and np.random.rand() < np.exp((new_sum - best_sum) / T)):
                curr_c = new_c
                curr_r = new_r
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_x[0::3] = curr_c[:, 0]
                    best_x[1::3] = curr_c[:, 1]
                    best_x[2::3] = curr_r
            
            T *= 0.996
            
            # Periodic SLSQP polish to refine locally
            if step % 40 == 0 and step > 0:
                x0 = best_x.copy()
                x0[2::3] *= 0.995
                x0 = project_to_feasible(x0)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                                   options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun):
                        c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_p = solve_lp_radii(c_p)
                        if np.sum(r_p) > best_sum:
                            best_sum = np.sum(r_p)
                            best_x[0::3] = c_p[:, 0]
                            best_x[1::3] = c_p[:, 1]
                            best_x[2::3] = r_p
                            curr_c, curr_r = c_p, r_p
                except Exception:
                    pass

    # Phase 4: Coordinate Descent Fine-Tuning
    if best_x is not None:
        curr_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        improved = True
        while improved:
            improved = False
            for i in range(N):
                best_local_sum = np.sum(best_x[2::3])
                for dx in [-0.004, -0.002, -0.001, 0.001, 0.002, 0.004]:
                    for dy in [-0.004, -0.002, -0.001, 0.001, 0.002, 0.004]:
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
            
        radii *= 0.9998
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(np.sum(radii))
