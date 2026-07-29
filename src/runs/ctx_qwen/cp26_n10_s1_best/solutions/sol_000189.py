# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000170 (state dbfe0634) state=2a819164 sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N = 26
TRIL_IDX = np.tril_indices(N, -1)
TRIU_IDX = np.triu_indices(N, 1)

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
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = max(0.0, d)
            idx += 1
            
    bounds = [(0.0, 0.5)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
    except Exception:
        pass
    return np.full(n, 0.02)

def center_penalty(c_flat, radii):
    """Smooth penalty function for centers given fixed radii."""
    c = c_flat.reshape(-1, 2)
    p = 0.0
    x, y, r = c[:, 0], c[:, 1], radii
    
    # Boundary penalties
    p += 100.0 * np.sum(np.maximum(r - x, 0.0)**2)
    p += 100.0 * np.sum(np.maximum(x - (1.0 - r), 0.0)**2)
    p += 100.0 * np.sum(np.maximum(r - y, 0.0)**2)
    p += 100.0 * np.sum(np.maximum(y - (1.0 - r), 0.0)**2)
    
    # Overlap penalties
    dx = c[:, 0, None] - c[:, 0]
    dy = c[:, 1, None] - c[:, 1]
    d = np.hypot(dx, dy)
    gap = d - r[:, None] - r[None, :]
    p += 500.0 * np.sum(np.maximum(-gap[TRIU_IDX], 0.0)**2)
    return p

def opt_centers_lbfgs(centers, radii):
    """Optimize centers for fixed radii using L-BFGS-B."""
    bounds = [(0.001, 0.999)] * (2 * N)
    res = minimize(center_penalty, centers.flatten(), args=(radii,),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15})
    return res.x.reshape(-1, 2)

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
        ca, sa = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
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

def constraints_slsqp(x):
    """Inequality constraints for SLSQP: g(x) >= 0."""
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    c = np.concatenate([c, np.hypot(dx[TRIL_IDX], dy[TRIL_IDX]) - dr[TRIL_IDX]])
    return c

def objective_func(x):
    """Objective: maximize sum of radii."""
    return -np.sum(x[2::3])

def basin_hop_search(c0, r0, steps=600):
    """Simulated annealing on centers using LP radii evaluation."""
    c = c0.copy()
    r = r0.copy()
    best_sum = np.sum(r)
    best_c = c.copy()
    best_r = r.copy()
    T = 0.04
    
    for _ in range(steps):
        c_new = c.copy()
        choice = np.random.rand()
        if choice < 0.6:
            idx = np.random.choice(N, np.random.randint(3, 10), replace=False)
            c_new[idx] += np.random.normal(0, 0.006, (len(idx), 2))
        elif choice < 0.8:
            ang = np.random.uniform(-0.04, 0.04)
            ca, sa = np.cos(ang), np.sin(ang)
            c_new = (c - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
        else:
            s = np.random.uniform(0.96, 1.04)
            c_new = (c - 0.5) * s + 0.5
            c_new += np.random.uniform(-0.015, 0.015, c.shape)
            
        c_new = np.clip(c_new, 0.005, 0.995)
        r_new = solve_lp_radii(c_new)
        s_new = np.sum(r_new)
        
        if s_new > best_sum or (T > 1e-5 and np.random.rand() < np.exp((s_new - best_sum) / T)):
            c, r = c_new, r_new
            if s_new > best_sum:
                best_sum, best_c, best_r = s_new, c_new.copy(), r_new.copy()
        T *= 0.994
        
    return best_c, best_r, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_obj = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_slsqp}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    for ang in np.linspace(-0.35, 0.35, 15):
        inits.append(make_hex_init(0.092, ang))
    for s in range(10):
        inits.append(make_force_init(s))
    for s in range(8):
        np.random.seed(s + 1000)
        inits.append(np.random.uniform(0.15, 0.85, (N, 2)))

    # Phase 2: Alternating Optimization + SLSQP Polish
    for pts in inits:
        c = pts.copy()
        r = solve_lp_radii(c)
        
        # Alternate LP radii and L-BFGS-B centers
        for _ in range(15):
            r = solve_lp_radii(c)
            c = opt_centers_lbfgs(c, r)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = np.maximum(r * 0.995, 1e-6)
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-14})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum and np.min(constraints_slsqp(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Basin Hopping to escape local minima
    if best_x is not None:
        bc = np.column_stack((best_x[0::3], best_x[1::3]))
        br = best_x[2::3]
        
        for _ in range(3):
            c_new, r_new, s_new = basin_hop_search(bc, br, steps=500)
            if s_new > best_sum:
                best_sum = s_new
                best_x = np.zeros(3 * N)
                best_x[0::3] = c_new[:, 0]
                best_x[1::3] = c_new[:, 1]
                best_x[2::3] = r_new
                bc, br = c_new, r_new
                
        # Final SLSQP polish after basin hopping
        x0 = best_x.copy()
        x0[2::3] *= 0.998
        for i in range(N):
            r = x0[3 * i + 2]
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                           options={'maxiter': 20000, 'ftol': 1e-14})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum and np.min(constraints_slsqp(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 4: Rotation Search & Deflation Repositioning
    if best_x is not None:
        bc = np.column_stack((best_x[0::3], best_x[1::3]))
        for ang_deg in np.linspace(-12, 12, 13):
            ang = np.deg2rad(ang_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            rot_c = (bc - 0.5) @ np.array([[ca, -sa], [sa, ca]]) + 0.5
            rot_c = np.clip(rot_c, 0.01, 0.99)
            r_rot = solve_lp_radii(rot_c)
            
            x_rot = np.zeros(3 * N)
            x_rot[0::3] = rot_c[:, 0]
            x_rot[1::3] = rot_c[:, 1]
            x_rot[2::3] = r_rot * 0.99
            try:
                res = minimize(objective_func, x_rot, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_r = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_r = solve_lp_radii(c_r)
                    if np.sum(r_r) > best_sum:
                        best_sum = np.sum(r_r)
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_r[:, 0]
                        best_x[1::3] = c_r[:, 1]
                        best_x[2::3] = r_r
            except Exception:
                pass
                
        # Aggressive deflation repositioning
        for cyc in range(40):
            x0 = best_x.copy()
            noise = 0.002 * (0.90 ** cyc)
            x0[0::3] += np.random.normal(0, noise, N)
            x0[1::3] += np.random.normal(0, noise, N)
            subset = np.random.choice(N, size=max(4, N // 3), replace=False)
            x0[subset * 3 + 2] *= 0.82
            for i in range(N):
                r = max(1e-6, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                x0[3 * i + 2] = r
            try:
                res = minimize(objective_func, x0, method='SLSQP', bounds=bounds_obj, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-13})
                if not np.isnan(res.fun):
                    c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(c_p)
                    if np.sum(r_p) > best_sum:
                        best_sum = np.sum(r_p)
                        best_x = np.zeros(3 * N)
                        best_x[0::3] = c_p[:, 0]
                        best_x[1::3] = c_p[:, 1]
                        best_x[2::3] = r_p
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
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-9:
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
