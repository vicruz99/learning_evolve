# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=ee948f59 sum of radii=2.475825 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def simulate(centers, r, steps=2000, dt=0.004, tol=1e-5):
    """Runs a force-directed repulsion simulation to check if a given radius r is feasible."""
    n = centers.shape[0]
    for _ in range(steps):
        forces = np.zeros_like(centers)
        
        # Boundary repulsion forces
        mask_left = centers[:, 0] < r
        forces[mask_left, 0] += (r - centers[mask_left, 0]) * 100.0
        mask_right = centers[:, 0] > 1.0 - r
        forces[mask_right, 0] -= (centers[mask_right, 0] - (1.0 - r)) * 100.0
        mask_bottom = centers[:, 1] < r
        forces[mask_bottom, 1] += (r - centers[mask_bottom, 1]) * 100.0
        mask_top = centers[:, 1] > 1.0 - r
        forces[mask_top, 1] -= (centers[mask_top, 1] - (1.0 - r)) * 100.0
        
        # Pairwise repulsion forces (vectorized)
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dist = np.sqrt(np.maximum(dx**2 + dy**2, 1e-12))
        np.fill_diagonal(dist, np.inf)
        
        min_dist = 2.0 * r
        overlap = np.maximum(0.0, min_dist - dist)
        rep_strength = (overlap * 50.0) / dist
        
        fx = np.sum(dx * rep_strength, axis=1)
        fy = np.sum(dy * rep_strength, axis=1)
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        centers += forces * dt
        centers = np.clip(centers, 0.0, 1.0)
        
    # Check maximum overlap after simulation
    max_overlap = 0.0
    max_overlap = max(max_overlap, np.max(r - centers[:, 0]), np.max(r - (1.0 - centers[:, 0])))
    max_overlap = max(max_overlap, np.max(r - centers[:, 1]), np.max(r - (1.0 - centers[:, 1])))
    
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    max_overlap = max(max_overlap, np.max(2.0 * r - d))
    
    return max_overlap < tol, centers

def s_obj(v, n):
    """Objective function for SLSQP: minimize negative radius (maximize r)."""
    return -v[-1]

def s_cons(v, n):
    """Constraint function for SLSQP: returns array of constraint values >= 0."""
    x = v[:-1:2]
    y = v[1::2]
    r = v[-1]
    c = []
    
    # Boundary constraints
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints
    xd = x[:, None] - x[None, :]
    yd = y[:, None] - y[None, :]
    dists = np.sqrt(xd**2 + yd**2)
    
    rows, cols = np.triu_indices(n, k=1)
    c.append(dists[rows, cols] - 2.0 * r)
    
    return np.concatenate(c)

def run_packing():
    n = 26
    configs = []
    
    # 1. Hexagonal lattice initialization
    hex_pts = []
    r_init = 0.08
    dy = r_init * np.sqrt(3.0)
    y = r_init
    row = 0
    while y < 1.0:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x < 1.0:
            hex_pts.append([x, y])
            x += 2.0 * r_init
        y += dy
        row += 1
    configs.append(np.array(hex_pts[:n]))
    
    # 2. Uniform grid initialization
    grid_pts = []
    step = 1.0 / 6.0
    for i in range(1, 7):
        for j in range(1, 7):
            if len(grid_pts) < n:
                grid_pts.append([i * step, j * step])
    configs.append(np.array(grid_pts))
    
    # 3. Perturbed hexagonal initialization
    np.random.seed(42)
    configs.append(np.clip(np.array(hex_pts[:n]) + np.random.uniform(-0.04, 0.04, (n, 2)), 0.1, 0.9))
    
    best_r = 0.0
    best_centers = None
    
    # Binary search for the maximum feasible radius
    low, high = 0.08, 0.115
    for _ in range(25):
        mid = (low + high) / 2.0
        feasible_any = False
        for cfg in configs:
            is_feas, final_centers = simulate(cfg, mid, steps=2000, dt=0.004, tol=1e-5)
            if is_feas:
                feasible_any = True
                if mid > best_r:
                    best_r = mid
                    best_centers = final_centers.copy()
                break  # Found a feasible configuration for this radius, try higher
        
        if feasible_any:
            low = mid
        else:
            high = mid
            
    # Refine the best configuration found using SLSQP for precise constraint satisfaction
    if best_centers is not None and best_r > 0:
        x0 = np.concatenate([best_centers.flatten(), [best_r]])
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]
        cons_dict = {'type': 'ineq', 'fun': s_cons, 'args': (n,)}
        
        res = minimize(s_obj, x0, args=(n,), method='SLSQP', bounds=bounds, 
                       constraints=cons_dict, options={'maxiter': 1500, 'ftol': 1e-12})
        
        if res.success:
            best_r = res.x[-1]
            best_centers = res.x[:-1].reshape(n, 2)
            
    radii = np.full(n, best_r)
    return best_centers, radii, best_r * n
