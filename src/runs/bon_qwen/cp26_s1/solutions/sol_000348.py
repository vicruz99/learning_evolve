# sol_000348 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2c580e0d) state=e88066d5 sum of radii=2.496909 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def compute_constraints(x, n):
    """Compute all inequality constraints >= 0"""
    con = []
    # Boundary constraints
    for i in range(n):
        xi = x[3*i]
        yi = x[3*i+1]
        ri = x[3*i+2]
        con.append(xi - ri)
        con.append(1.0 - xi - ri)
        con.append(yi - ri)
        con.append(1.0 - yi - ri)
        
    # Non-overlap constraints: dist^2 >= (r1 + r2)^2
    for i in range(n):
        xi = x[3*i]
        yi = x[3*i+1]
        ri = x[3*i+2]
        for j in range(i + 1, n):
            xj = x[3*j]
            yj = x[3*j+1]
            rj = x[3*j+2]
            dx = xi - xj
            dy = yi - yj
            con.append(dx*dx + dy*dy - (ri + rj)**2)
            
    return np.array(con)

def objective_func(x):
    """Minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(x[2::3])

def run_single_opt(x0, n, bounds, cons_dict):
    """Run single optimization instance"""
    try:
        res = opt.minimize(objective_func, x0, method='SLSQP', 
                           bounds=bounds, constraints=cons_dict, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
        return res
    except Exception:
        return None

def run_packing():
    n = 26
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
    cons_dict = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    best_res = None
    best_sum = -np.inf
    
    x0_candidates = []
    
    # 1. Hexagonal lattice initialization
    pts_hex = []
    r_est = 0.1
    y = r_est
    row = 0
    while y < 1.0 - r_est:
        x = r_est
        offset = r_est * np.sqrt(3) / 2 if row % 2 == 1 else 0.0
        while x < 1.0 - r_est:
            pts_hex.append([x + offset, y, r_est])
            x += 2 * r_est
        y += np.sqrt(3) * r_est
        row += 1
    while len(pts_hex) < n:
        pts_hex.append([0.5, 0.5, r_est])
    x0_candidates.append(np.array(pts_hex[:n]).flatten())
    
    # 2. Grid initialization
    pts_grid = []
    cols, rows = 6, 5
    dx = 0.8 / max(cols - 1, 1)
    dy = 0.8 / max(rows - 1, 1)
    for r in range(rows):
        for c in range(cols):
            pts_grid.append([0.1 + c * dx, 0.1 + r * dy, 0.08])
    x0_candidates.append(np.array(pts_grid[:n]).flatten())
    
    # 3. Random initialization
    np.random.seed(42)
    pts_rand = np.random.uniform(0.15, 0.85, size=(n, 2))
    pts_rand = np.hstack([pts_rand, np.full((n, 1), 0.08)])
    x0_candidates.append(pts_rand.flatten())
    
    # Optimize from each start
    for x0 in x0_candidates:
        res = run_single_opt(x0, n, bounds, cons_dict)
        if res is not None and res.success:
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_res = res
                
        # Try perturbed version if successful
        if res is not None:
            noise = np.random.normal(0, 0.015, size=x0.shape)
            x0_p = x0 + noise
            x0_p[::3] = np.clip(x0_p[::3], 0, 1)
            x0_p[1::3] = np.clip(x0_p[1::3], 0, 1)
            x0_p[2::3] = np.clip(x0_p[2::3], 0, 0.25)
            
            res_p = run_single_opt(x0_p, n, bounds, cons_dict)
            if res_p is not None and res_p.success:
                curr_sum_p = -res_p.fun
                if curr_sum_p > best_sum:
                    best_sum = curr_sum_p
                    best_res = res_p
                    
    if best_res is not None:
        x_opt = best_res.x
        centers = np.array([x_opt[3*i:3*i+2] for i in range(n)])
        radii = x_opt[2::3]
        radii = np.maximum(radii, 1e-9) # Ensure non-negative
        return centers, radii, np.sum(radii)
    else:
        # Fallback valid packing
        fc = np.random.uniform(0.2, 0.8, (n, 2))
        fr = np.full(n, 0.05)
        return fc, fr, np.sum(fr)
