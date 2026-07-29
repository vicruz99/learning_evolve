# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000083 (state dd0fff3c) state=abae6780 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints g(x) >= 0."""
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
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
            
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def make_hex_init(r0, angle=0.0):
    """Generates a hexagonal lattice initialization with optional rotation."""
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
        if y > 1.0 + r0: break
            
    pts = np.array(pts[:N + 10])
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_directed_init(seed, steps=400):
    """Force-directed layout to spread points evenly."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    for _ in range(steps):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[j] - pts[i]
                d = np.linalg.norm(dx)
                if d < 0.3 and d > 1e-5:
                    f = 0.01 / (d**2 + 0.001)
                    forces[i] -= f * dx
                    forces[j] += f * dx
            for dim in range(2):
                if pts[i, dim] < 0.15: forces[i, dim] += 0.02
                elif pts[i, dim] > 0.85: forces[i, dim] -= 0.02
        pts += forces * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def project_to_bounds(x0):
    """Ensure optimization vector strictly respects bounds."""
    x0 = x0.copy()
    for i in range(N):
        r = max(1e-7, x0[3 * i + 2])
        x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
        x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
        x0[3 * i + 2] = r
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_x = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}

    # Phase 1: Diverse Initial Configurations
    configs = []
    # Hexagonal lattices with varied scales and rotations
    for scale in [0.95, 1.0, 1.05, 1.1]:
        r0 = 0.09 * scale
        for ang in np.linspace(-0.3, 0.3, 9):
            configs.append(make_hex_init(r0, ang))
            
    # Force-directed layouts
    for s in range(6):
        configs.append(force_directed_init(s))
        
    # Structured Grid
    c_grid = np.zeros((N, 2))
    idx = 0
    for i in range(6):
        for j in range(5):
            if idx < N:
                c_grid[idx] = [0.12 + j * 0.18, 0.12 + i * 0.16]
                idx += 1
    configs.append(c_grid)

    # Phase 2: Multi-start Optimization with LP-SLSQP Alternation
    for c_init in configs:
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        
        # Initial LP snap
        r_lp = solve_lp_radii(c_init)
        if r_lp is None:
            r_lp = np.full(N, 0.06)
        x0[2::3] = r_lp
        x0 = project_to_bounds(x0)

        # Alternating refinement cycles
        for _ in range(4):
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    x0 = res.x.copy()
                    # LP snap on new centers
                    centers_new = np.column_stack((x0[0::3], x0[1::3]))
                    r_new = solve_lp_radii(centers_new)
                    if r_new is not None:
                        x0[2::3] = r_new
                        x0 = project_to_bounds(x0)
            except Exception:
                break

        curr_sum = np.sum(x0[2::3])
        if curr_sum > best_sum:
            if np.min(constraints(x0)) >= -1e-7:
                best_sum = curr_sum
                best_x = x0.copy()

    # Phase 3: Aggressive Topology Search (Deflation, Swapping, Rotation)
    if best_x is not None:
        # 3a: Adaptive Noise Deflation
        for step in range(30):
            noise = 0.002 * (0.92 ** step)
            x_p = best_x.copy()
            x_p += np.random.normal(0, noise, 3 * N)
            x_p[2::3] *= 0.985
            x_p = project_to_bounds(x_p)

            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    centers_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(centers_p)
                    if r_p is not None:
                        x_p[2::3] = r_p
                        x_p = project_to_bounds(x_p)
                        curr_sum = np.sum(x_p[2::3])
                        if curr_sum > best_sum and np.min(constraints(x_p)) >= -1e-7:
                            best_sum = curr_sum
                            best_x = x_p.copy()
            except Exception:
                pass

        # 3b: Subset Deflation (Break local lattice structures)
        for _ in range(25):
            x_p = best_x.copy()
            subset = np.random.choice(N, size=N // 3, replace=False)
            x_p[2::3][subset] *= 0.6
            x_p = project_to_bounds(x_p)
            
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    centers_p = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_p = solve_lp_radii(centers_p)
                    if r_p is not None:
                        x_p[2::3] = r_p
                        x_p = project_to_bounds(x_p)
                        curr_sum = np.sum(x_p[2::3])
                        if curr_sum > best_sum and np.min(constraints(x_p)) >= -1e-7:
                            best_sum = curr_sum
                            best_x = x_p.copy()
            except Exception:
                pass

        # 3c: Global Rotation Search
        best_centers = np.column_stack((best_x[0::3], best_x[1::3]))
        best_radii = best_x[2::3]
        for ang_deg in np.linspace(-10, 10, 11):
            ang = np.deg2rad(ang_deg)
            c, s = np.cos(ang), np.sin(ang)
            rot_centers = (best_centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
            
            # Clamp to keep inside
            rot_centers = np.clip(rot_centers, 0.01, 0.99)
            
            r_rot = solve_lp_radii(rot_centers)
            if r_rot is not None and np.sum(r_rot) > best_sum:
                x_rot = np.zeros(3 * N)
                x_rot[0::3] = rot_centers[:, 0]
                x_rot[1::3] = rot_centers[:, 1]
                x_rot[2::3] = r_rot
                x_rot = project_to_bounds(x_rot)
                
                try:
                    res = minimize(objective, x_rot, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    if not np.isnan(res.fun):
                        centers_rot = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_final = solve_lp_radii(centers_rot)
                        if r_final is not None:
                            x_rot[2::3] = r_final
                            x_rot = project_to_bounds(x_rot)
                            curr_sum = np.sum(x_rot[2::3])
                            if curr_sum > best_sum and np.min(constraints(x_rot)) >= -1e-7:
                                best_sum = curr_sum
                                best_x = x_rot.copy()
                except Exception:
                    pass

    # Fallback (should not be reached)
    if best_x is None:
        best_x = np.zeros(3 * N)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 6), 5)[:N]
        best_x[2::3] = 0.06

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()

    # Final strict validity adjustment against 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0: valid = False; break
            if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > 1.0 + 1e-9: valid = False; break
            if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > 1.0 + 1e-9: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9: valid = False; break
                if not valid: break
        if valid: break
        
        # Minimal shrinkage to recover strict feasibility
        radii *= 0.9999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    return centers, radii, float(np.sum(radii))
