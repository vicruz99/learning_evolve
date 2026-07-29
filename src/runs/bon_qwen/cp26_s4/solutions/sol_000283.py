# sol_000283 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 20c39dac) state=a745dcf8 sum of radii=2.630060 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    # Maximize sum of radii => minimize negative sum
    # vars layout: [x0, y0, r0, x1, y1, r1, ...]
    return -np.sum(vars[2::3])

def constraints(vars):
    arr = vars.reshape(-1, 3)
    cx = arr[:, 0]
    cy = arr[:, 1]
    cr = arr[:, 2]
    
    cons = []
    # Boundary constraints: x >= r, y >= r, x + r <= 1, y + r <= 1
    cons.extend(cx - cr)
    cons.extend(cy - cr)
    cons.extend(1.0 - cx - cr)
    cons.extend(1.0 - cy - cr)
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(N_CIRCLES):
        dx = cx[i+1:] - cx[i]
        dy = cy[i+1:] - cy[i]
        dr = cr[i+1:] + cr[i]
        dists = np.sqrt(dx**2 + dy**2)
        cons.extend(dists - dr)
        
    return np.array(cons)

def run_packing():
    # Bounds for x, y in [0, 1] and r in [epsilon, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N_CIRCLES
    
    best_res = None
    best_val = -np.inf
    
    # Multi-start optimization to escape local minima
    for seed in range(20):
        np.random.seed(seed)
        # Initial grid placement approximating hexagonal packing
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        grid_pts = [(x, y) for x in xs for y in ys]
        grid_pts.append((0.5, 0.5))  # 26th circle
        np.random.shuffle(grid_pts)
        
        cx0 = np.array([p[0] for p in grid_pts]) + np.random.randn(N_CIRCLES) * 0.02
        cy0 = np.array([p[1] for p in grid_pts]) + np.random.randn(N_CIRCLES) * 0.02
        cr0 = np.ones(N_CIRCLES) * 0.08
        
        x0 = np.column_stack([cx0, cy0, cr0]).flatten()
        
        # Ensure initial guess respects bounds
        x0[0::3] = np.clip(x0[0::3], 0.0, 1.0)
        x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
        x0[2::3] = np.clip(x0[2::3], 1e-7, 0.5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 5000, 'ftol': 1e-12})
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_res = res
        except Exception:
            continue
            
    if best_res is None:
        # Fallback to a valid but suboptimal packing if optimization fails
        centers = np.column_stack([np.linspace(0.1, 0.9, 5) for _ in range(5)])[:N_CIRCLES]
        radii = np.full(N_CIRCLES, 0.05)
        return centers, radii, np.sum(radii)
        
    centers = np.column_stack([best_res.x[0::3], best_res.x[1::3]])
    radii = best_res.x[2::3]
    
    return centers, radii, np.sum(radii)
