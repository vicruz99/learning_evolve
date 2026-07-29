# sol_000107 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82d73ba2) state=58b7d739 sum of radii=2.319861 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # --- 1. Initialization ---
    # Generate a hexagonal lattice initial guess
    # We use a small radius to ensure validity, optimizer will grow them.
    r_init = 0.04
    s = 2 * r_init
    dy = s * np.sqrt(3) / 2
    dx = s
    
    points = []
    y = r_init
    row = 0
    while y + r_init <= 1.0:
        x = r_init
        if row % 2 == 1:
            x += dx / 2
        
        while x + r_init <= 1.0:
            points.append((x, y))
            x += dx
        
        y += dy
        row += 1
    
    # We need exactly n points. 
    # If we generated more, take the first n (which are well-distributed).
    # If fewer (unlikely with r=0.04), we would need to adjust, but 0.04 fits ~50+ circles.
    if len(points) < n:
        # Fallback to grid if hex generation fails (unlikely)
        step = 0.2
        pts = []
        for r_idx in range(6):
            for c_idx in range(6):
                pts.append((c_idx * step + 0.1, r_idx * step + 0.1))
        points = pts[:n]

    initial_centers = np.array(points[:n])
    initial_radii = np.full(n, r_init)
    
    # Flatten variables for scipy: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i]   = initial_centers[i, 0]
        x0[3*i+1] = initial_centers[i, 1]
        x0[3*i+2] = initial_radii[i]
    
    # Bounds: x,y in [0, 1], r in [0, 1] (loosely, actually r <= 0.5)
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    # --- 2. Optimization Setup ---
    
    def objective(v):
        # Minimize negative sum of radii
        radii = v[2::3]
        return -np.sum(radii)

    def constraints_func(v):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Unpack
        centers[:, 0] = v[0::3]
        centers[:, 1] = v[1::3]
        radii = v[2::3]
        
        # 1. Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
        # x >= r  => x - r >= 0
        # x <= 1-r => 1 - x - r >= 0
        c_bounds = np.concatenate([
            centers[:, 0] - radii,
            1.0 - centers[:, 0] - radii,
            centers[:, 1] - radii,
            1.0 - centers[:, 1] - radii
        ])
        
        # 2. Distance constraints: dist^2 >= (r_i + r_j)^2
        # Vectorized computation
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2) # shape (n, n)
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        # Constraint: dist_sq - r_sum_sq >= 0
        # We only need upper triangle (i < j)
        triu_indices = np.triu_indices(n, k=1)
        c_dist = dist_sq[triu_indices] - r_sum_sq[triu_indices]
        
        return np.concatenate([c_bounds, c_dist])

    constraint = {
        'type': 'ineq',
        'fun': constraints_func
    }

    # --- 3. Run Optimizer ---
    # Method SLSQP is suitable for this
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraint,
            options={'ftol': 1e-12, 'maxiter': 500, 'disp': False}
        )
        
        if res.success or res.fun < -2.0: # Check if it found something reasonable
            final_x = res.x
        else:
            # Fallback to initial guess if optimization fails completely
            final_x = x0
            
    except Exception:
        final_x = x0

    # --- 4. Extract and Return Results ---
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_x[3*i]
        final_centers[i, 1] = final_x[3*i+1]
        final_radii[i] = final_x[3*i+2]
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
