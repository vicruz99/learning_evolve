# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state cc21d5f7) state=b05dbff6 sum of radii=2.561297 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def constraints(vars_vec):
    """Compute all inequality constraints: boundary and separation."""
    n = 26
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c_list = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise separation constraints: dist_sq >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c_list.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c_list)

def spread_centers(pts, steps=150):
    """Force-directed layout to spread points evenly in the square."""
    pts = pts.copy().astype(float)
    n = len(pts)
    for _ in range(steps):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.rand(2) * 1e-6
                rep = 1.0 / (dist**2)
                forces[i] += diff / dist * rep
                forces[j] -= diff / dist * rep
                
            # Soft wall repulsion to keep points inside [0.05, 0.95]
            for d in range(2):
                if pts[i, d] < 0.05:
                    forces[i, d] += (0.05 - pts[i, d]) * 20.0
                elif pts[i, d] > 0.95:
                    forces[i, d] -= (pts[i, d] - 0.95) * 20.0
                    
        pts += forces * 0.02
        pts = np.clip(pts, 0.01, 0.99)
    return pts

def run_packing():
    n = 26
    best_sum = -np.inf
    best_x = None
    
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Generate diverse initial center configurations
    configs = []
    
    # Config 1: Hexagonal lattice
    r_hex = 0.08
    pts_hex = []
    y = r_hex
    row = 0
    while len(pts_hex) < n:
        x = r_hex if row % 2 == 0 else 2 * r_hex
        while x <= 1.0 - r_hex and len(pts_hex) < n:
            pts_hex.append([x, y])
            x += 2 * r_hex
        y += np.sqrt(3) * r_hex
        row += 1
    configs.append(np.array(pts_hex[:n]))
    
    # Config 2: 5x5 Grid + Center
    pts_grid = []
    for i in range(5):
        for j in range(5):
            pts_grid.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    pts_grid.append([0.5, 0.5])
    configs.append(np.array(pts_grid[:n]))
    
    # Config 3-9: Random + Force Spreading
    for seed in range(7):
        np.random.seed(seed + 200)
        pts_rand = np.random.rand(n, 2)
        pts_rand = spread_centers(pts_rand, steps=150)
        configs.append(pts_rand)
        
    # Main optimization loop
    for idx, pts in enumerate(configs):
        # Compute strictly feasible initial radii
        r_init = np.full(n, 0.001)
        for i in range(n):
            max_r = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            for j in range(n):
                if i == j: continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d < max_r * 2:
                    max_r = d / 2.0
            r_init[i] = max_r * 0.90  # 10% safety margin ensures strict feasibility
            
        # Flatten to optimization vector [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = pts[i, 0]
            x0[3*i+1] = pts[i, 1]
            x0[3*i+2] = r_init[i]
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 4000, 'ftol': 1e-12})
            
            # Check feasibility tolerance
            if res.success or (np.min(constraints(res.x)) >= -1e-8):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is not None:
        # Perturbation refinement: escape shallow local minima
        best_x = best_x.copy()
        for _ in range(6):
            x_pert = best_x + np.random.randn(3*n) * 0.004
            # Project back to bounds
            for i in range(n):
                x_pert[3*i] = np.clip(x_pert[3*i], 0.0, 1.0)
                x_pert[3*i+1] = np.clip(x_pert[3*i+1], 0.0, 1.0)
                x_pert[3*i+2] = np.clip(x_pert[3*i+2], 0.0, 0.5)
                
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraints},
                               options={'maxiter': 2500, 'ftol': 1e-12})
                if np.min(constraints(res.x)) >= -1e-8:
                    curr = -res.fun
                    if curr > best_sum:
                        best_sum = curr
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Extract final centers and radii
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    return centers, radii, float(best_sum)
