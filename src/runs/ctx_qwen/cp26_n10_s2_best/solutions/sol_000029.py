# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=af044a19 sum of radii=2.630818 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars[2*n:])

def constraint_func(vars, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = []
    cons.append(centers[:, 0] - radii)
    cons.append(1.0 - centers[:, 0] - radii)
    cons.append(centers[:, 1] - radii)
    cons.append(1.0 - centers[:, 1] - radii)
    
    # Non-overlap constraints: dist >= r_i + r_j
    c_i = centers[pair_i]
    c_j = centers[pair_j]
    dists = np.sqrt(np.sum((c_i - c_j)**2, axis=1))
    r_sum = radii[pair_i] + radii[pair_j]
    cons.append(dists - r_sum)
    
    return np.concatenate(cons)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n, pair_i, pair_j)}
    
    best_sum = -1.0
    best_x = None
    
    inits = []
    
    # 1. Hexagonal lattice configurations
    for shift in [0.0, 0.02, -0.02, 0.04]:
        r0 = 0.085
        y = r0 + shift
        row = 0
        pts = []
        while len(pts) < n:
            x_start = r0 if row % 2 == 0 else 2 * r0
            x = x_start
            while x <= 1 - r0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        inits.append(np.array(pts[:n]))
        
    # 2. Square grid configurations
    for shift in [0.0, 0.03, -0.02]:
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([i/4.0 + 0.05 + shift, j/4.0 + 0.05 + shift])
        pts.append([0.5, 0.5])
        inits.append(np.array(pts[:n]))
        
    # 3. Dense row pattern configurations
    patterns = [[6,5,6,5,4], [7,5,6,5,3], [6,6,5,5,4], [5,6,5,6,4]]
    for pat in patterns:
        pts = []
        ry = 0.07
        for r_idx, count in enumerate(pat):
            rx = 0.07 if r_idx % 2 == 0 else 0.14
            for c in range(count):
                pts.append([rx + c * 0.15, ry])
            ry += 0.16
        inits.append(np.array(pts[:n]))
        
    # 4. Random configurations
    for seed in range(12):
        np.random.seed(seed)
        inits.append(np.random.rand(n, 2) * 0.8 + 0.1)
        
    # 5. Perturbed center-cluster configs
    for seed in range(5):
        np.random.seed(200 + seed)
        # Place 4 large-ish circles in corners, fill rest randomly
        pts = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]])
        rest = np.random.rand(n-4, 2) * 0.6 + 0.2
        pts = np.vstack([pts, rest])
        inits.append(pts)

    # Optimization loop
    for base_centers in inits:
        for pert_seed in range(4):
            np.random.seed(pert_seed + 500)
            centers_pert = base_centers + np.random.uniform(-0.015, 0.015, size=base_centers.shape)
            centers_pert = np.clip(centers_pert, 0.03, 0.97)
            radii_init = np.full(n, 0.04)
            x0 = np.concatenate([centers_pert.flatten(), radii_init])
            
            res = minimize(objective_func, x0, args=(n,), method='SLSQP', 
                           bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                           
            if -res.fun > best_sum:
                c_sol = res.x[:2*n].reshape(n, 2)
                r_sol = res.x[2*n:]
                
                # Strict validity check
                valid = True
                if np.any(c_sol[:, 0] - r_sol < -1e-7) or np.any(c_sol[:, 0] + r_sol > 1.0 + 1e-7) or \
                   np.any(c_sol[:, 1] - r_sol < -1e-7) or np.any(c_sol[:, 1] + r_sol > 1.0 + 1e-7):
                    valid = False
                if valid:
                    d = np.sqrt(np.sum((c_sol[pair_i] - c_sol[pair_j])**2, axis=1))
                    if np.any(d < r_sol[pair_i] + r_sol[pair_j] - 1e-7):
                        valid = False
                if valid:
                    best_sum = -res.fun
                    best_x = res.x.copy()
                    
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Final safety adjustment to guarantee strict validator compliance
    radii *= 0.999995
    centers = np.clip(centers, 1e-8, 1.0 - 1e-8)
    
    return centers, radii, float(np.sum(radii))
