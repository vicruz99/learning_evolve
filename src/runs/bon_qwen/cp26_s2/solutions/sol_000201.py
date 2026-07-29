# sol_000201 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=b0f296e7 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # --- Initialization ---
    # Start with a hexagonal grid pattern, which is a dense packing structure.
    # Initial radius guess 0.09 allows us to fit 26 circles comfortably.
    r_init = 0.09
    centers_list = []
    dy = r_init * np.sqrt(3)
    dx = 2 * r_init
    
    j = 0
    while True:
        y = r_init + j * dy
        if y > 1 - r_init:
            break
        offset = (j % 2) * r_init
        i = 0
        while True:
            x = r_init + offset + i * dx
            if x > 1 - r_init:
                break
            centers_list.append([x, y])
            i += 1
        j += 1
        
    centers_list = np.array(centers_list)
    
    # Fallback if hex grid is sparse (unlikely with r=0.09)
    if len(centers_list) < n:
        pts = []
        step = 0.15
        for i in range(8):
            for j in range(8):
                pts.append([i*step + step/2, j*step + step/2])
        centers_list = np.array(pts)
        r_init = 0.07
        
    # Select n centers
    init_centers = centers_list[:n]
    
    # Add small random noise to break symmetry and help local search
    np.random.seed(42)
    noise = np.random.uniform(-0.005, 0.005, size=init_centers.shape)
    init_centers += noise
    # Clip to ensure valid range for initial radius
    init_centers = np.clip(init_centers, r_init, 1.0 - r_init)
    
    # Flatten variables: [x1, y1, r1, ..., xn, yn, rn]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = init_centers[i, 0]
        x0[3*i+1] = init_centers[i, 1]
        x0[3*i+2] = r_init
        
    # --- Optimization ---
    
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(vars[2::3])
        
    def all_constraints(vars):
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        # Inequality constraints must be >= 0
        con_boundary = np.concatenate([
            xs - rs,
            1.0 - xs - rs,
            ys - rs,
            1.0 - ys - rs
        ])
        
        # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # Vectorized computation
        diff_x = xs[:, None] - xs[None, :]
        diff_y = ys[:, None] - ys[None, :]
        dist_sq = diff_x**2 + diff_y**2
        
        sum_r = rs[:, None] + rs[None, :]
        sum_r_sq = sum_r**2
        
        diff = dist_sq - sum_r_sq
        
        # Extract upper triangle (i < j) to avoid duplicates and self-checks
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        con_pairs = diff[mask]
        
        return np.concatenate([con_boundary, con_pairs])

    cons_dict = {'type': 'ineq', 'fun': all_constraints}
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Run SLSQP optimizer
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-12})
    
    x_opt = res.x
    
    # --- Post-processing ---
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = x_opt[3*i]
        centers[i, 1] = x_opt[3*i+1]
        radii[i] = max(0.0, x_opt[3*i+2])
        
    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r_bound = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r_bound:
            radii[i] = max_r_bound
            
    # Resolve overlaps by shrinking radii
    # Iterate to resolve transitive overlaps
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    overlap = sum_r - dist
                    reduction = overlap / 2.0
                    radii[i] -= reduction
                    radii[j] -= reduction
                    if radii[i] < 0: radii[i] = 0
                    if radii[j] < 0: radii[j] = 0
                    changed = True
        if not changed:
            break
            
    return centers, radii, np.sum(radii)
