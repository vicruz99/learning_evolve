# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000036 (state 025191a3) state=14b2ae40 sum of radii=2.629619 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint_func(vars):
    """Vectorized inequality constraints: g(vars) >= 0."""
    n = N_CIRCLES
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    b = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 - (r1+r2)^2 >= 0
    dx = x[:, None] - x
    dy = y[:, None] - y
    dr = r[:, None] + r
    
    i, j = np.tril_indices(n, -1)
    o = dx[i, j]**2 + dy[i, j]**2 - dr[i, j]**2
    
    return np.concatenate([b, o])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [1e-7, 0.5]."""
    b = []
    for _ in range(N_CIRCLES):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def get_safe_radii(centers):
    """Compute strictly feasible initial radii based on nearest neighbors/walls."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                if d < min_d:
                    min_d = d
        # Leave 55% margin to guarantee initial feasibility for optimizer
        radii[i] = min_d * 0.45
    return radii

def generate_hex_points(n, r0, angle_deg):
    """Generate hexagonal lattice points, optionally rotated."""
    pts = []
    y = r0
    row = 0
    while len(pts) < n + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        if y > 1.0 + r0:
            break
            
    pts = np.array(pts[:n + 10])
    
    # Rotate around center (0.5, 0.5)
    if angle_deg != 0:
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        center = np.array([0.5, 0.5])
        pts = (pts - center) @ np.array([[c, -s], [s, c]]) + center
        
    # Filter points strictly inside the square
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    # Pad if rotation/filtering removed too many points
    while len(pts) < n:
        pts = np.vstack([pts, [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]])
        
    return pts[:n]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    n = N_CIRCLES
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    inits = []
    
    # 1. Rotated Hexagonal Lattices (varied densities & angles)
    for r0 in [0.08, 0.09, 0.10, 0.11]:
        for angle in [0, 10, 20, 30, 45]:
            pts = generate_hex_points(n, r0, angle)
            pts += np.random.uniform(-0.003, 0.003, pts.shape)
            inits.append(pts.copy())
            
    # 2. Perturbed Grids
    for _ in range(6):
        pts = []
        for r in range(6):
            for c in range(5):
                if len(pts) >= n: break
                pts.append([0.08 + c * 0.18, 0.08 + r * 0.16])
        pts = np.array(pts[:n]) + np.random.uniform(-0.012, 0.012, (n, 2))
        inits.append(pts.copy())
        
    # 3. Force-Directed Layouts (captures non-lattice structures)
    for _ in range(5):
        pts = np.random.uniform(0.15, 0.85, (n, 2))
        for _ in range(800):
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pts[j, 0] - pts[i, 0]
                    dy = pts[j, 1] - pts[i, 1]
                    d = np.hypot(dx, dy)
                    if d < 0.28 and d > 1e-6:
                        f = 0.015 / (d**2)
                        fx, fy = f * dx, f * dy
                        forces[i] -= [fx, fy]
                        forces[j] += [fx, fy]
            for i in range(n):
                if pts[i, 0] < 0.12: forces[i, 0] += 0.025
                elif pts[i, 0] > 0.88: forces[i, 0] -= 0.025
                if pts[i, 1] < 0.12: forces[i, 1] += 0.025
                elif pts[i, 1] > 0.88: forces[i, 1] -= 0.025
            pts += forces * 0.06
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts.copy())

    # Phase 1: Multi-start optimization
    for pts in inits:
        radii = get_safe_radii(pts)
        x0 = np.zeros(3 * n)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = radii
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = res.x[2::3]
                
                # Strict feasibility validation
                valid = True
                for i in range(n):
                    if r_opt[i] < 1e-8 or c_opt[i,0] < r_opt[i] - 1e-9 or c_opt[i,0] > 1 - r_opt[i] + 1e-9 or \
                       c_opt[i,1] < r_opt[i] - 1e-9 or c_opt[i,1] > 1 - r_opt[i] + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = np.hypot(c_opt[i,0] - c_opt[j,0], c_opt[i,1] - c_opt[j,1])
                            if d < r_opt[i] + r_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid: break
                
                if valid and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue

    # Phase 2: Deflation-Reoptimization Refinement
    # Shrinking radii slightly allows centers to move into denser configurations
    if best_centers is not None:
        for round in range(12):
            pert_c = best_centers + np.random.normal(0, 0.0015, best_centers.shape)
            pert_c = np.clip(pert_c, 0.01, 0.99)
            pert_r = best_radii * 0.985
            x0 = np.zeros(3 * n)
            x0[0::3] = pert_c[:, 0]
            x0[1::3] = pert_c[:, 1]
            x0[2::3] = pert_r
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_opt = res.x[2::3]
                        
                        valid = True
                        for i in range(n):
                            if r_opt[i] < 1e-8 or c_opt[i,0] < r_opt[i] - 1e-9 or c_opt[i,0] > 1 - r_opt[i] + 1e-9 or \
                               c_opt[i,1] < r_opt[i] - 1e-9 or c_opt[i,1] > 1 - r_opt[i] + 1e-9:
                                valid = False
                                break
                        if valid:
                            for i in range(n):
                                for j in range(i + 1, n):
                                    d = np.hypot(c_opt[i,0] - c_opt[j,0], c_opt[i,1] - c_opt[j,1])
                                    if d < r_opt[i] + r_opt[j] - 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                        if valid and curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = c_opt.copy()
                            best_radii = r_opt.copy()
            except Exception:
                pass

    # Fallback (should rarely be reached)
    if best_centers is None:
        best_centers = generate_hex_points(n, 0.09, 0)
        best_radii = get_safe_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
