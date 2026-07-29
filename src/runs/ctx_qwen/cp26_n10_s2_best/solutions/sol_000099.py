# sol_000099 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state af044a19) state=eca92967 sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n, pi, pj):
    """Compute inequality constraints: boundaries and non-overlap (squared)."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    cons = np.empty(4*n + len(pi))
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons[:n] = c[:, 0] - r
    cons[n:2*n] = 1.0 - c[:, 0] - r
    cons[2*n:3*n] = c[:, 1] - r
    cons[3*n:4*n] = 1.0 - c[:, 1] - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = c[pi, 0] - c[pj, 0]
    dy = c[pi, 1] - c[pj, 1]
    dist_sq = dx**2 + dy**2
    r_sum = r[pi] + r[pj]
    cons[4*n:] = dist_sq - r_sum**2
    
    return cons

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pi, pj = np.triu_indices(n, k=1)
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n, pi, pj)}
    
    best_val = -np.inf
    best_x = None
    
    # Collect diverse initial configurations
    starts = []
    
    # 1. Hexagonal lattice variations
    for seed in range(15):
        np.random.seed(seed)
        r0 = 0.09 + np.random.uniform(-0.01, 0.01)
        pts = []
        y = r0
        row = 0
        while len(pts) < n:
            x = r0 if row % 2 == 0 else 2 * r0
            while x <= 1.0 - r0 and len(pts) < n:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:n])
        pts += np.random.uniform(-0.015, 0.015, pts.shape)
        pts = np.clip(pts, 0.03, 0.97)
        starts.append(np.concatenate([pts.flatten(), np.full(n, 0.05)]))
        
    # 2. Random uniform starts
    for seed in range(15):
        np.random.seed(seed + 100)
        pts = np.random.uniform(0.1, 0.9, (n, 2))
        starts.append(np.concatenate([pts.flatten(), np.full(n, 0.04)]))
        
    # 3. Grid-like starts with jitter
    for seed in range(10):
        np.random.seed(seed + 200)
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08 + i * 0.16, 0.08 + j * 0.18])
        pts = np.array(pts[:n])
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        starts.append(np.concatenate([pts.flatten(), np.full(n, 0.05)]))

    # Primary optimization pass
    for x0 in starts:
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                c_val = constraints(res.x, n, pi, pj)
                if np.min(c_val) >= -1e-8:
                    best_val = -res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is None:
        best_x = starts[0]
        best_val = -objective(best_x, n)

    # Local refinement: perturb and re-optimize to escape local minima
    current_x = best_x
    for step in range(10):
        scale = 0.98 - step * 0.005
        pert_x = current_x.copy()
        pert_x[2*n:] *= scale  # Shrink radii to create breathing room
        
        noise = np.random.uniform(-0.01, 0.01, 2*n)
        pert_x[:2*n] += noise
        pert_x[:2*n] = np.clip(pert_x[:2*n], 0.02, 0.98)
        
        try:
            res = minimize(objective, pert_x, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                c_val = constraints(res.x, n, pi, pj)
                if np.min(c_val) >= -1e-8:
                    best_val = -res.fun
                    best_x = res.x.copy()
                    current_x = best_x
        except Exception:
            continue
            
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
