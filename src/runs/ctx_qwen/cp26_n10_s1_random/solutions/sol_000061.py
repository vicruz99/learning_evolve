# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000010 (state f39c4564) state=9a59633b sum of radii=2.604587 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(x):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(x[2::3])

def constraint_func(x, n):
    """
    Computes inequality constraints.
    Returns array of values that must be >= 0 for a valid packing.
    Uses squared distances for smooth gradients.
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints (4n)
    b_cons = np.concatenate([
        cx - r,              # x - r >= 0
        1.0 - cx - r,        # 1 - x - r >= 0
        cy - r,              # y - r >= 0
        1.0 - cy - r         # 1 - y - r >= 0
    ])
    
    # Pairwise non-overlap constraints (n*(n-1)/2)
    n_pairs = n * (n - 1) // 2
    p_cons = np.empty(n_pairs)
    idx = 0
    for i in range(n):
        xi, yi, ri = cx[i], cy[i], r[i]
        for j in range(i + 1, n):
            dx = xi - cx[j]
            dy = yi - cy[j]
            # Squared constraint: dist^2 - (r_i + r_j)^2 >= 0
            p_cons[idx] = dx*dx + dy*dy - (ri + r[j])**2
            idx += 1
            
    return np.concatenate([b_cons, p_cons])

def run_packing():
    n = 26
    
    # Bounds: x,y in [0, 1], r in [1e-6, 0.5]
    bounds = []
    for k in range(3 * n):
        if k % 3 == 2:
            bounds.append((1e-6, 0.5))
        else:
            bounds.append((0.0, 1.0))
            
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    best_sum = 0.0
    best_x = None
    
    # --- Generate Initial Configurations ---
    configs = []
    
    # 1. Hexagonal lattice initialization
    r_init = 0.075
    pts = []
    y = r_init
    row = 0
    while y + r_init < 1.0:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init < 1.0:
            pts.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
    pts = np.array(pts[:n])
    configs.append(pts)
    
    # 2. Structured row pattern (5-6-5-6-4) closely matching known optimal topology
    r_str = 0.075
    pat = []
    y_curr = r_str
    dy = np.sqrt(3) * r_str
    rows_def = [(5, 0.0), (6, r_str), (5, 0.0), (6, r_str), (4, 0.0)]
    
    for count, shift in rows_def:
        if len(pat) >= n: break
        # Evenly space centers within valid x-range, apply hex shift
        x_start = r_str + shift
        x_end = 1.0 - r_str
        if count > 1:
            xs = np.linspace(x_start, x_end, count)
        else:
            xs = np.array([0.5])
        for x in xs:
            if len(pat) < n:
                pat.append([x, y_curr])
        y_curr += dy
    if len(pat) >= n:
        configs.append(np.array(pat[:n]))
        
    # 3. Randomized perturbations to escape local minima
    np.random.seed(42)
    for _ in range(10):
        noise = np.random.uniform(-0.03, 0.03, size=(n, 2))
        cfg = np.clip(configs[0] + noise, 0.05, 0.95)
        configs.append(cfg)
        
    # --- Optimization Loop ---
    for centers_init in configs:
        x0 = np.empty(3 * n)
        x0[0::3] = centers_init[:, 0]
        x0[1::3] = centers_init[:, 1]
        x0[2::3] = 0.075  # Start feasible
        
        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-11, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                s = np.sum(res.x[2::3])
                # Verify constraint satisfaction with tolerance
                c_vals = constraint_func(res.x, n)
                if np.min(c_vals) >= -1e-5:
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # --- Final Extraction & Safety Scaling ---
    if best_x is not None:
        centers = np.column_stack((best_x[0::3], best_x[1::3]))
        radii = best_x[2::3].copy()
        
        # Compute maximum safe scaling factor to guarantee strict validity
        min_scale = 1.0
        
        # Boundary limits
        for i in range(n):
            r = radii[i]
            if r < 1e-9: continue
            min_scale = min(min_scale, 
                centers[i,0]/r, (1.0-centers[i,0])/r,
                centers[i,1]/r, (1.0-centers[i,1])/r)
                
        # Pairwise limits
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                r_sum = radii[i] + radii[j]
                if r_sum < 1e-9: continue
                ratio = d / r_sum
                if ratio < min_scale:
                    min_scale = ratio
                    
        # Apply slight shrinkage to absorb floating-point inaccuracies
        # Validator allows 1e-12 tolerance, so 0.999999 is safe and preserves precision
        scale_factor = min_scale * 0.999999
        radii *= scale_factor
        best_sum = np.sum(radii)
        
        return centers, radii, best_sum
    else:
        # Fallback to a valid hex grid if optimization fails
        centers = configs[0]
        radii = np.full(n, 0.075)
        return centers, radii, np.sum(radii)
