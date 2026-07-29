# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000014 (state d34ac82b) state=cf517c54 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def constr_func(x, n):
    """Inequality constraints: boundary and pairwise non-overlap."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    c = []
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for all pairs
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    c_pairwise = dist_sq - r_sum**2
    
    # Extract upper triangle (i < j) to avoid duplicates and self-pairs
    idx = np.triu_indices(n, k=1)
    c.extend(c_pairwise[idx])
    
    return np.array(c)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constr_func, 'args': (n,)}
    
    best_sum = 0.0
    best_x = None
    
    # 1. Create a base hexagonal layout for good structural initialization
    r_init = 0.095
    s = 2.0 * r_init
    base_pts = []
    y = r_init + 0.01
    row = 0
    while len(base_pts) < n:
        shift = s / 2 if row % 2 == 1 else 0
        x = r_init + 0.01 + shift
        while x <= 1 - r_init - 0.01 and len(base_pts) < n:
            base_pts.append((x, y))
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    base_pts = np.array(base_pts[:n])
    
    # 2. Multiple restarts to explore the configuration space
    num_restarts = 50
    for trial in range(num_restarts):
        rng = np.random.RandomState(trial * 13 + 7)
        
        if trial < 30:
            # Perturbed hexagonal lattice
            p = base_pts + rng.normal(0, 0.015, base_pts.shape)
            p = np.clip(p, 0.05, 0.95)
        else:
            # Random placement in safe inner region
            p = rng.uniform(0.12, 0.88, (n, 2))
            
        # Initialize radii slightly smaller to ensure initial feasibility
        r0 = 0.07 + rng.uniform(0, 0.02)
        x0 = np.concatenate([p[:, 0], p[:, 1], np.full(n, r0)])
        
        try:
            res = minimize(
                obj_func, x0, args=(n,),
                method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # 3. Local refinement: perturb best solution to escape local minima
    if best_x is not None:
        for _ in range(15):
            rng = np.random.RandomState(_ * 17 + 3)
            x_pert = best_x + rng.normal(0, 0.003, len(best_x))
            x_pert = np.clip(x_pert, [b[0] for b in bounds], [b[1] for b in bounds])
            
            res = minimize(
                obj_func, x_pert, args=(n,),
                method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
            )
            if res.success and -res.fun > best_sum:
                best_sum = -res.fun
                best_x = res.x.copy()
                
    # 4. Extract results
    if best_x is None:
        # Fallback to base hex if optimization completely fails
        best_x = np.concatenate([base_pts[:, 0], base_pts[:, 1], np.full(n, 0.08)])
        
    cx = best_x[:n]
    cy = best_x[n:2*n]
    r = best_x[2*n:]
    
    centers = np.column_stack((cx, cy))
    radii = np.maximum(r, 0.0)
    
    # 5. Post-processing: fix tiny numerical violations to guarantee validity
    for _ in range(200):
        changed = False
        # Boundary violations
        for i in range(n):
            max_r = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
            if radii[i] > max_r + 1e-10:
                radii[i] = max(0.0, max_r - 1e-10)
                changed = True
                
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < radii[i] + radii[j] - 1e-10:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess / 2.0
                    radii[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
