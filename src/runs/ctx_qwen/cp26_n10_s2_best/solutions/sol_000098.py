# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state af044a19) state=47444e7d sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraint_func(v, n, pair_i, pair_j):
    """
    Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for better numerical stability and differentiability.
    Returns array where all elements >= 0 indicates a satisfied constraint.
    """
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    cons_size = 4 * n + len(pair_i)
    cons = np.empty(cons_size)
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    idx = 0
    cons[idx:idx+n] = centers[:, 0] - radii
    idx += n
    cons[idx:idx+n] = 1.0 - centers[:, 0] - radii
    idx += n
    cons[idx:idx+n] = centers[:, 1] - radii
    idx += n
    cons[idx:idx+n] = 1.0 - centers[:, 1] - radii
    idx += n
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = centers[pair_i, 0] - centers[pair_j, 0]
    dy = centers[pair_i, 1] - centers[pair_j, 1]
    rs = radii[pair_i] + radii[pair_j]
    cons[idx:] = dx**2 + dy**2 - rs**2
    
    return cons

def generate_initial_configs(n):
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying spacing and shifts
    for scale in np.linspace(0.075, 0.105, 6):
        for sx in np.linspace(-0.03, 0.03, 3):
            for sy in np.linspace(-0.02, 0.02, 3):
                pts = []
                y = sy + scale
                row = 0
                while len(pts) < n + 5:
                    x_start = sx + (scale if row % 2 == 0 else 2 * scale)
                    x = x_start
                    while x <= 1.0 - scale and len(pts) < n + 5:
                        pts.append([x, y])
                        x += 2 * scale
                    y += scale * np.sqrt(3)
                    row += 1
                if len(pts) >= n:
                    configs.append(np.array(pts[:n]))
                    
    # 2. Square grids with varying spacing
    for spacing in np.linspace(0.11, 0.15, 5):
        pts = []
        y = spacing
        while len(pts) < n:
            x = spacing
            while x <= 1.0 - spacing and len(pts) < n:
                pts.append([x, y])
                x += spacing
            y += spacing
        if len(pts) >= n:
            configs.append(np.array(pts[:n]))
            
    # 3. Random placements
    np.random.seed(123)
    for _ in range(8):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n, pair_i, pair_j)}
    opt_options = {'maxiter': 6000, 'ftol': 1e-12, 'disp': False}
    
    best_sum = -1.0
    best_v = None
    
    initial_configs = generate_initial_configs(n)
    
    # Phase 1: Multi-start optimization
    for i, centers in enumerate(initial_configs):
        # Add slight jitter to break symmetry
        centers_j = centers + np.random.uniform(-0.005, 0.005, size=centers.shape)
        centers_j = np.clip(centers_j, 0.05, 0.95)
        
        # Feasible initial radii
        r_init = np.full(n, 0.04)
        v0 = np.concatenate([centers_j.flatten(), r_init])
        
        try:
            res = minimize(objective_func, v0, args=(n,), method='SLSQP', bounds=bounds, 
                           constraints=cons_dict, options=opt_options)
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Strict feasibility check
                c_vals = constraint_func(res.x, n, pair_i, pair_j)
                if np.min(c_vals) > -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_v is None:
        best_v = np.zeros(3 * n)
        best_v[:2 * n] = np.random.uniform(0.2, 0.8, 2 * n)
        best_v[2 * n:] = 0.05
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(20):
        # Perturb centers
        perturbation = np.random.uniform(-0.004, 0.004, size=2 * n)
        v_pert = current_v.copy()
        v_pert[:2 * n] += perturbation
        v_pert[:2 * n] = np.clip(v_pert[:2 * n], 0.02, 0.98)
        
        # Slightly shrink radii to guarantee feasibility after perturbation
        v_pert[2 * n:] *= 0.985
        
        try:
            res = minimize(objective_func, v_pert, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options=opt_options)
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_vals = constraint_func(res.x, n, pair_i, pair_j)
                if np.min(c_vals) > -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    centers = best_v[:2 * n].reshape(n, 2)
    radii = best_v[2 * n:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(15):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d:
                    excess = radii[i] + radii[j] - d
                    shrink = excess / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
