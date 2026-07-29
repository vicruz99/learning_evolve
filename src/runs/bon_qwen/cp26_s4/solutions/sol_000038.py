# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 12653929) state=b2146365 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def objective(v):
        return -np.sum(v[2::3])
        
    def all_constraints(v):
        cons = []
        for i in range(n):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            cons.append(xi - ri)
            cons.append((1.0 - xi) - ri)
            cons.append(yi - ri)
            cons.append((1.0 - yi) - ri)
            cons.append(ri)
            
        for i in range(n):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dx, dy = xi - xj, yi - yj
                dist_sq = dx*dx + dy*dy
                r_sum = ri + rj
                cons.append(dist_sq - r_sum*r_sum)
        return np.array(cons)

    cons = {'type': 'ineq', 'fun': all_constraints}
    bounds = [(0, 1) if k % 3 != 2 else (0, 0.5) for k in range(3*n)]
    
    best_v = None
    best_val = -np.inf
    
    configs = []
    
    # 1. 5x5 Grid + 1
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append(((i+0.5)/5.0, (j+0.5)/5.0))
    pts.append((0.5, 0.5))
    configs.append(pts)
    
    # 2. Hexagonal Lattice
    hex_pts = []
    y_inc = np.sqrt(3)/2 * 0.22
    x_inc = 0.22
    for row in range(7):
        for col in range(5):
            x = col * x_inc + 0.11
            y = row * y_inc + 0.11 + (col % 2) * y_inc / 2
            if x <= 1.0 and y <= 1.0:
                hex_pts.append((x, y))
        if len(hex_pts) >= n: break
    configs.append(hex_pts[:n])
    
    # 3. Random
    np.random.seed(42)
    configs.append([(np.random.rand(), np.random.rand()) for _ in range(n)])
    
    # 4. Concentric Rings
    ring_pts = []
    for k in range(1, 4):
        m = 10 * k
        for i in range(m):
            angle = 2 * np.pi * i / m
            r_center = 0.25 * k
            ring_pts.append((0.5 + r_center * np.cos(angle), 0.5 + r_center * np.sin(angle)))
        if len(ring_pts) >= n: break
    configs.append(ring_pts[:n])

    for cfg in configs:
        while len(cfg) < n:
            cfg.append((np.random.rand(), np.random.rand()))
        cfg = cfg[:n]
        
        v0 = np.array([])
        for p in cfg:
            v0 = np.append(v0, [p[0], p[1], 0.05])
            
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
            
            c_vals = all_constraints(res.x)
            if np.min(c_vals) >= -1e-7:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        v0 = np.array([])
        for i in range(5):
            for j in range(5):
                v0 = np.append(v0, [(i+0.5)/5.0, (j+0.5)/5.0, 0.05])
        v0 = np.append(v0, [0.5, 0.5, 0.05])
        best_v = v0

    centers = best_v.reshape(n, 3)[:, :2]
    radii = best_v.reshape(n, 3)[:, 2]
    
    # Clamp to boundaries strictly
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1-r)
        
    return centers, radii, np.sum(radii)
