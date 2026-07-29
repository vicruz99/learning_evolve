# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 62f34940) state=84168184 sum of radii=2.614639 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

def get_hex_init(n, spacing=0.14, shift=0.05):
    """Generate hexagonal lattice initialization."""
    pts = []
    y = spacing
    row = 0
    while len(pts) < n:
        x = spacing if row % 2 == 0 else spacing + shift
        while x <= 1.0 - spacing and len(pts) < n:
            pts.append([x, y])
            x += 2 * spacing
        y += spacing * math.sqrt(3) / 2
        row += 1
    return np.array(pts[:n])

def get_grid_init(n, cols=5, rows=6):
    """Generate structured grid initialization."""
    pts = []
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    for r in range(rows):
        for c in range(cols):
            if len(pts) < n:
                pts.append([dx * (c + 1), dy * (r + 1)])
    return np.array(pts[:n])

def solve_radii_lp(centers, n):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Pairwise non-overlap: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            d = math.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Boundary constraints handled via variable bounds
    bounds_r = []
    for i in range(n):
        mx = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        bounds_r.append((0.0, max(0.0, mx - 1e-11)))

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # 1. Generate diverse initial configurations
    configs = []
    for sp in [0.11, 0.13, 0.15, 0.17]:
        for sh in [0.0, 0.02, 0.05]:
            configs.append(get_hex_init(n, sp, sh))
    configs.append(get_grid_init(n, 5, 6))
    configs.append(get_grid_init(n, 6, 5))
    configs.append(get_grid_init(n, 7, 4))
    for _ in range(6):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))

    # SLSQP objective and constraints
    def obj(x):
        return -np.sum(x[2*n:])

    def constr(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        c = []
        # Boundary constraints
        c.extend(cx - r)
        c.extend(1.0 - cx - r)
        c.extend(cy - r)
        c.extend(1.0 - cy - r)
        # Overlap constraints (squared to avoid sqrt singularity)
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx*dx + dy*dy
                rs = r[i] + r[j]
                c.append(d2 - rs*rs)
        return np.array(c)

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constr}

    # 2. Optimize from each initialization
    for cfg in configs:
        r0 = np.full(n, 0.015)
        for i in range(n):
            mx = min(cfg[i,0], 1.0-cfg[i,0], cfg[i,1], 1.0-cfg[i,1])
            r0[i] = min(r0[i], mx * 0.9)

        x0 = np.concatenate([cfg[:,0], cfg[:,1], r0])

        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 25000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cx = res.x[:n]
                cy = res.x[n:2*n]
                centers = np.column_stack((cx, cy))
                radii, s = solve_radii_lp(centers, n)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers
                    best_radii = radii
        except Exception:
            continue

    # 3. Perturbation & Local Search refinement
    if best_centers is not None:
        for _ in range(8):
            pert = best_centers + np.random.normal(0, 0.003, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            r0 = best_radii * 0.95
            x0 = np.concatenate([pert[:,0], pert[:,1], r0])
            try:
                res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                               options={'maxiter': 20000, 'ftol': 1e-14})
                if res.success:
                    cx = res.x[:n]
                    cy = res.x[n:2*n]
                    centers = np.column_stack((cx, cy))
                    radii, s = solve_radii_lp(centers, n)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers
                        best_radii = radii
            except Exception:
                pass

    # 4. Fallback if all optimizations failed
    if best_centers is None or np.any(np.isnan(best_centers)) or np.any(np.isnan(best_radii)):
        best_centers = get_grid_init(n, 5, 6)
        best_radii, best_sum = solve_radii_lp(best_centers, n)

    return best_centers, best_radii, float(best_sum)
