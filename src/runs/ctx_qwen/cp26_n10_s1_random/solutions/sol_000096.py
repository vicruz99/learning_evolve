# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000045 (state 7c76ac7a) state=22c44e92 sum of radii=1.443347 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Vectorized inequality constraints >= 0 for valid packing."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, dist[mask] - r_sum[mask]])
    return c

def simulate(centers, radii, steps=1200):
    """Force-directed simulation to resolve overlaps and expand circles."""
    n = N
    c = centers.copy()
    r = radii.copy()
    dt = 0.004
    rep = 150.0
    wall = 100.0
    
    for _ in range(steps):
        forces = np.zeros((n, 2))
        
        # Boundary repulsion
        left_ov = np.maximum(0.0, r - c[:,0])
        right_ov = np.maximum(0.0, c[:,0] + r - 1.0)
        bottom_ov = np.maximum(0.0, r - c[:,1])
        top_ov = np.maximum(0.0, c[:,1] + r - 1.0)
        
        forces[:,0] += wall * (left_ov - right_ov)
        forces[:,1] += wall * (bottom_ov - top_ov)
        
        # Pairwise repulsion
        diff = c[:, None, :] - c[None, :, :]  # Points from i to j
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        req = r[:, None] + r[None, :]
        overlap = np.maximum(0.0, req - dist_mat)
        overlap[dist_mat < 1e-8] = 0.0
        
        inv_dist = np.where(dist_mat > 1e-8, 1.0/dist_mat, 0.0)
        dir_x = diff[:,:,0] * inv_dist
        dir_y = diff[:,:,1] * inv_dist
        
        # Push i away from j
        forces[:,0] -= np.sum(overlap * dir_x * rep, axis=1)
        forces[:,1] -= np.sum(overlap * dir_y * rep, axis=1)
        
        c += forces * dt
        c = np.clip(c, 0.005, 0.995)
        r += 0.000015
        r = np.clip(r, 0.05, 0.5)
        
    return c, r

def run_packing():
    n = N
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    configs = []
    
    # 1. Hexagonal lattice initialization
    r0 = 0.10
    pts = []
    y = r0
    row = 0
    while y < 1.0 and len(pts) < n:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    base_hex = np.array(pts[:n])
    configs.append(base_hex)
    
    # 2. Perturbed hex grids
    for i in range(14):
        cfg = base_hex + np.random.uniform(-0.025, 0.025, (n, 2))
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    # 3. Random uniform configurations
    for i in range(8):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # Optimization loop over all configurations
    for cfg in configs:
        # Pre-condition with force simulation
        c_sim, r_sim = simulate(cfg, np.full(n, 0.08), steps=1000)
        
        x0 = np.zeros(3*n)
        x0[0::3] = c_sim[:,0]
        x0[1::3] = c_sim[:,1]
        x0[2::3] = r_sim
        
        # Constrained optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 3500, 'ftol': 1e-14})
                       
        centers = res.x[:2*n].reshape(n, 2)
        
        # Post-process: compute exact maximal feasible radii for these centers
        radii = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            min_d = min(centers[i,0], 1.0-centers[i,0], 
                        centers[i,1], 1.0-centers[i,1])
            # Distance to other circles
            dists = np.sqrt(np.sum((centers - centers[i])**2, axis=1))
            dists[i] = np.inf
            min_d = min(min_d, np.min(dists) / 2.0)
            radii[i] = min_d
            
        # Validate and apply safety buffer
        if np.all(radii > 1e-6):
            radii *= 0.9999999  # Tiny margin for 1e-12 grader tolerance
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Fallback (should not trigger given robust init)
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (n, 2))
        best_radii = np.full(n, 0.05)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
