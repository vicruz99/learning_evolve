# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000073 (state 134a5cab) state=42a15999 sum of radii=2.624702 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Vectorized inequality constraints: g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - cr,
        1.0 - cx - cr,
        cy - cr,
        1.0 - cy - cr
    ])
    
    # Overlap constraints using linear distances for better conditioning
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    i, j = np.tril_indices(N_CIRCLES, -1)
    dist = np.hypot(dx[i, j], dy[i, j])
    c = np.concatenate([c, dist - dr[i, j]])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N_CIRCLES):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], 
                  centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(1e-9, lim)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(1e-9, d)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def generate_init_configs():
    """Generate diverse initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Rotated Hexagonal Lattices
    for r0 in [0.08, 0.09, 0.10, 0.11]:
        for ang in [0.0, 0.1, -0.1, 0.2, -0.2, 0.35]:
            pts = []
            y = r0
            row = 0
            while len(pts) < N_CIRCLES + 6:
                x = r0 if row % 2 == 0 else 2.0 * r0
                while x <= 1.0 - r0:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3.0) * r0
                row += 1
                if y > 1.0 + r0:
                    break
                    
            pts = np.array(pts[:N_CIRCLES + 6])
            if ang != 0.0:
                c, s = np.cos(ang), np.sin(ang)
                center = np.array([0.5, 0.5])
                pts = (pts - center) @ np.array([[c, -s], [s, c]]) + center
                
            mask = (pts[:, 0] >= 0.02) & (pts[:, 0] <= 0.98) & \
                   (pts[:, 1] >= 0.02) & (pts[:, 1] <= 0.98)
            pts = pts[mask]
            
            if len(pts) < N_CIRCLES:
                pad = N_CIRCLES - len(pts)
                pts = np.vstack([pts, np.random.uniform(0.1, 0.9, (pad, 2))])
            configs.append(pts[:N_CIRCLES].copy())
            
    # 2. Corner/Edge Focused Layouts
    for _ in range(10):
        pts = np.zeros((N_CIRCLES, 2))
        r_corner = 0.10
        pts[0] = [r_corner, r_corner]
        pts[1] = [1.0 - r_corner, r_corner]
        pts[2] = [r_corner, 1.0 - r_corner]
        pts[3] = [1.0 - r_corner, 1.0 - r_corner]
        
        idx = 4
        y = 0.25
        row = 0
        while idx < N_CIRCLES:
            x = 0.25 if row % 2 == 0 else 0.35
            while x <= 0.80 and idx < N_CIRCLES:
                pts[idx] = [x + np.random.uniform(-0.02, 0.02), 
                            y + np.random.uniform(-0.02, 0.02)]
                idx += 1
                x += 0.18
            y += 0.15
            row += 1
        configs.append(pts.copy())
        
    # 3. Force-Directed Random Layouts
    for _ in range(8):
        pts = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
        for _ in range(300):
            forces = np.zeros_like(pts)
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    dx = pts[j] - pts[i]
                    d = np.hypot(dx[0], dx[1])
                    if d < 0.3 and d > 1e-6:
                        f = 0.005 / (d**2)
                        forces[i] -= f * dx
                        forces[j] += f * dx
            for i in range(N_CIRCLES):
                if pts[i, 0] < 0.15: forces[i, 0] += 0.02
                elif pts[i, 0] > 0.85: forces[i, 0] -= 0.02
                if pts[i, 1] < 0.15: forces[i, 1] += 0.02
                elif pts[i, 1] > 0.85: forces[i, 1] -= 0.02
            pts += forces * 0.03
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts.copy())
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    inits = generate_init_configs()
    
    # Phase 1: Multi-start Optimization
    for pts in inits:
        radii = np.full(N_CIRCLES, 0.06)
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = radii
        
        # Ensure initial strict feasibility
        for i in range(N_CIRCLES):
            r = radii[i]
            x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 15000, 
                                                      'ftol': 1e-14, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                centers = np.column_stack((res.x[0::3], res.x[1::3]))
                lp_radii = solve_lp_radii(centers)
                if lp_radii is not None:
                    res.x[2::3] = lp_radii
                    curr_sum = np.sum(lp_radii)
                    
                if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-7:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Deflation & Perturbation Refinement
    if best_x is not None:
        for step in range(25):
            x0 = best_x.copy()
            # Shrink radii to free space for repositioning
            x0[2::3] *= 0.975
            noise_scale = 0.002 * (1.0 - step / 25.0)
            x0[0::3] += np.random.normal(0, noise_scale, N_CIRCLES)
            x0[1::3] += np.random.normal(0, noise_scale, N_CIRCLES)
            
            # Project back to feasible bounds
            for i in range(N_CIRCLES):
                r = max(0.005, x0[3 * i + 2])
                x0[3 * i] = np.clip(x0[3 * i], r, 1.0 - r)
                x0[3 * i + 1] = np.clip(x0[3 * i + 1], r, 1.0 - r)
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints=cons, options={'maxiter': 10000, 
                                                          'ftol': 1e-14, 'disp': False})
                if not np.isnan(res.fun):
                    curr_sum = -res.fun
                    centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    lp_radii = solve_lp_radii(centers)
                    if lp_radii is not None:
                        res.x[2::3] = lp_radii
                        curr_sum = np.sum(lp_radii)
                        
                    if curr_sum > best_sum and np.min(constraints(res.x)) >= -1e-7:
                        best_sum = curr_sum
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Extract results
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final safety adjustment to guarantee strict compliance with 1e-12 tolerance
    for _ in range(100):
        valid = True
        for i in range(N_CIRCLES):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-12 or \
               centers[i, 0] > 1.0 - radii[i] + 1e-12 or \
               centers[i, 1] < radii[i] - 1e-12 or \
               centers[i, 1] > 1.0 - radii[i] + 1e-12:
                valid = False
                break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    d = np.hypot(centers[i, 0] - centers[j, 0], 
                                 centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
            
        # Minimal shrinkage to recover validity
        radii *= 0.9995
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
