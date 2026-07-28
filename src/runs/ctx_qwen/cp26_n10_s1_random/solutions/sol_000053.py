# sol_000053 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state 6f2d6856) state=2b1a7146 sum of radii=2.257815 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars):
    """Computes all boundary and pairwise non-overlap constraints as a single vector."""
    n = 26
    X = vars.reshape(n, 3)
    cx = X[:, 0]
    cy = X[:, 1]
    r = X[:, 2]
    
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    b_cons = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    pair_cons = dist_sq - r_sum**2
    
    # Extract upper triangle to avoid duplicates and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_flat = pair_cons[mask].flatten()
    
    return np.concatenate([b_cons, p_flat])

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    constraint_dict = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate initial configurations
    inits = []
    
    # Base hexagonal lattice
    r0 = 0.08
    pts = []
    y = r0
    row = 0
    while y + r0 <= 1.0:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x + r0 <= 1.0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        if len(pts) >= n:
            break
            
    while len(pts) < n:
        pts.append([0.5, 0.5])
        
    hex_base = np.array(pts[:n])
    inits.append(hex_base)
    
    # Perturbed hex grids to escape local minima
    np.random.seed(123)
    for _ in range(14):
        noise = np.random.uniform(-0.06, 0.06, size=(n, 2))
        p = hex_base + noise
        p = np.clip(p, 0.05, 0.95)
        inits.append(p)
        
    # Run optimization from each starting configuration
    for centers_init in inits:
        x0 = np.zeros(3 * n)
        x0[0::3] = centers_init[:, 0]
        x0[1::3] = centers_init[:, 1]
        x0[2::3] = 0.05  # Start with small feasible radii
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraint_dict,
                options={'ftol': 1e-13, 'maxiter': 3000, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 3)[:, :2]
                r_opt = res.x.reshape(n, 3)[:, 2]
                
                # Strict validation matching the grader's tolerance
                valid = True
                for i in range(n):
                    if (c_opt[i, 0] < r_opt[i] - 1e-12 or c_opt[i, 0] + r_opt[i] > 1.0 + 1e-12 or 
                        c_opt[i, 1] < r_opt[i] - 1e-12 or c_opt[i, 1] + r_opt[i] > 1.0 + 1e-12):
                        valid = False
                        break
                        
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = np.sqrt(np.sum((c_opt[i] - c_opt[j])**2))
                            if d < r_opt[i] + r_opt[j] - 1e-12:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
                        
        except Exception:
            continue
            
    # Fallback to a valid hex grid if optimization yields nothing
    if best_centers is None:
        best_centers = hex_base
        best_radii = np.full(n, 0.09)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
