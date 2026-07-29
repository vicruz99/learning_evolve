# sol_000293 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e8a963c) state=094c46ee sum of radii=2.499330 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(params, n):
    """Objective function: maximize sum of radii with penalties for violations."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Boundary penalty: r_i must be <= dist to each edge
    bounds = np.zeros((n, 4))
    bounds[:, 0] = centers[:, 0]          # left
    bounds[:, 1] = 1.0 - centers[:, 0]   # right
    bounds[:, 2] = centers[:, 1]          # bottom
    bounds[:, 3] = 1.0 - centers[:, 1]   # top
    b_pen = np.sum(np.maximum(0, radii[:, None] - bounds)**2)
    
    # Overlap penalty: r_i + r_j <= d_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dists[mask]
    r_sum = radii[:, None] + radii[None, :]
    r_sum = r_sum[mask]
    o_pen = np.sum(np.maximum(0, r_sum - dists)**2)
    
    # Weighted penalty method
    return -np.sum(radii) + 5e5 * (b_pen + o_pen)

def relax_radii(centers, radii, n):
    """Iteratively shrink radii to guarantee strict non-overlap and boundary satisfaction."""
    radii = radii.copy()
    for _ in range(30):
        for i in range(n):
            max_r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                    cand = d - radii[j]
                    if cand < max_r:
                        max_r = cand
            radii[i] = max(0, max_r)
    return radii

def run_packing():
    n = 26
    best_obj = np.inf
    best_params = None
    
    inits = []
    
    # 1. Grid initialization
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 6))
    inits.append(np.column_stack([gx.ravel()[:n], gy.ravel()[:n]]))
    
    # 2. Hexagonal lattice initialization
    hex_pts = []
    row = 0
    y = 0.1
    r_h = 0.1
    while len(hex_pts) < n:
        x = 0.1 if row % 2 == 0 else 0.2
        while x <= 0.9 and len(hex_pts) < n:
            hex_pts.append([x, y])
            x += 2 * r_h
        y += np.sqrt(3) * r_h
        row += 1
    inits.append(np.array(hex_pts[:n]))
    
    # 3. Random initialization
    np.random.seed(42)
    inits.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    bounds_opt = [(0, 1)] * (2*n) + [(1e-5, 0.5)] * n
    
    # Multi-restart optimization
    for init_centers in inits:
        radii = np.full(n, 0.08)
        params0 = np.concatenate([init_centers.ravel(), radii])
        res = minimize(compute_loss, params0, args=(n,), method='L-BFGS-B', bounds=bounds_opt,
                       options={'maxiter': 15000, 'ftol': 1e-12})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
            
    # Final refinement pass
    res = minimize(compute_loss, best_params, args=(n,), method='L-BFGS-B', bounds=bounds_opt,
                   options={'maxiter': 20000, 'ftol': 1e-12})
    centers = res.x[:2*n].reshape(n, 2)
    radii = res.x[2*n:]
    
    # Guarantee strict validity
    radii = relax_radii(centers, radii, n)
    
    return centers, radii, np.sum(radii)
