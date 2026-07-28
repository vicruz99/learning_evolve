# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000010 (state f39c4564) state=1a94ab0e sum of radii=2.565995 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n):
    """Objective: maximize sum of radii => minimize negative sum"""
    return -np.sum(x[2::3])

def compute_constraints(x, n):
    """Constraints: boundary inclusion and non-overlap (squared)"""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints: x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0
    b_cons = np.concatenate([
        cx - r, 
        1.0 - cx - r, 
        cy - r, 
        1.0 - cy - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 - (ri+rj)^2 >= 0
    n_pairs = n * (n - 1) // 2
    p_cons = np.empty(n_pairs)
    idx = 0
    for i in range(n):
        xi, yi, ri = cx[i], cy[i], r[i]
        for j in range(i + 1, n):
            dx = xi - cx[j]
            dy = yi - cy[j]
            p_cons[idx] = dx*dx + dy*dy - (ri + r[j])**2
            idx += 1
            
    return np.concatenate([b_cons, p_cons])

def run_packing() -> tuple:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds for [x, y, r] for each circle
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n
    
    # Constraint definition
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice patterns with various shifts
    for sx in np.linspace(0.0, 0.12, 4):
        for sy in np.linspace(0.0, 0.12, 3):
            pts = []
            r0 = 0.09
            dy = r0 * np.sqrt(3)
            y = r0 + sy
            row = 0
            while y + r0 < 1.0 and len(pts) < n:
                x = r0 + sx
                if row % 2 == 1:
                    x += r0
                while x + r0 < 1.0 and len(pts) < n:
                    pts.append([x, y])
                    x += 2 * r0
                y += dy
                row += 1
            if len(pts) == n:
                configs.append(np.array(pts))
                
    # 2. Dense grid base
    gx = np.linspace(0.15, 0.85, 5)
    gy = np.linspace(0.15, 0.85, 5)
    grid_pts = np.array([[x, y] for y in gy for x in gx])[:25]
    grid_pts = np.vstack([grid_pts, [0.5, 0.5]])
    configs.append(grid_pts)
    
    # 3. Perturbed grids
    np.random.seed(42)
    for _ in range(4):
        noise = np.random.uniform(-0.06, 0.06, (n, 2))
        cfg = np.clip(grid_pts + noise, 0.1, 0.9)
        configs.append(cfg)
        
    # Run optimization for each configuration
    for pts in configs:
        x0 = np.empty(3 * n)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = 0.07  # Start with strictly feasible radii
        
        try:
            res = minimize(
                compute_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                args=(n,),
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                # Check constraint satisfaction
                cons_vals = compute_constraints(res.x, n)
                if np.all(cons_vals >= -1e-6):
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt = res.x[2::3]
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt
                        best_radii = r_opt
        except Exception:
            pass

    # Fallback to a known valid hexagonal packing
    if best_centers is None or best_sum < 2.0:
        pts = []
        r_fb = 0.08
        dy = r_fb * np.sqrt(3)
        y = r_fb
        row = 0
        while len(pts) < n and y + r_fb < 1.0:
            x = r_fb
            if row % 2 == 1:
                x += r_fb
            while x + r_fb < 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r_fb
            y += dy
            row += 1
        best_centers = np.array(pts[:n])
        best_radii = np.full(n, r_fb)
        best_sum = np.sum(best_radii)

    # Final safety scaling to guarantee strict validity
    x_final = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii])
    for _ in range(200):
        if np.min(compute_constraints(x_final, n)) >= -1e-9:
            break
        best_radii *= 0.9999
        x_final[2::3] = best_radii
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
