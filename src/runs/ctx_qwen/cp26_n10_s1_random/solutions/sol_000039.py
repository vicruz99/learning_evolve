# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=a2a59673 sum of radii=0.909999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars_array):
    """Computes the negative sum of radii plus smooth penalty terms for overlaps and boundaries."""
    cx = vars_array[0::3]
    cy = vars_array[1::3]
    r = vars_array[2::3]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty weight
    mu = 10000.0
    
    # Boundary penalties: circle must be inside [0,1]x[0,1]
    obj += mu * np.sum(np.maximum(0, r - cx)**2)
    obj += mu * np.sum(np.maximum(0, r + cx - 1.0)**2)
    obj += mu * np.sum(np.maximum(0, r - cy)**2)
    obj += mu * np.sum(np.maximum(0, r + cy - 1.0)**2)
    
    # Overlap penalties: dist(i,j) >= r_i + r_j
    diff_x = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    np.fill_diagonal(dist, 1.0) # Avoid zeros on diagonal for numerical stability
    
    r_sum = r[:, None] + r[None, :]
    viol = np.maximum(0, r_sum - dist)
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    obj += mu * 2.0 * np.sum((viol * mask)**2)
    
    return obj

def get_initial_config(method='hex'):
    """Generates an initial feasible configuration of centers and radii."""
    n = N_CIRCLES
    v = np.zeros(3*n)
    v[2::3] = 0.085  # Initial small radius to guarantee feasibility
    
    if method == 'hex':
        idx = 0
        y = 0.085
        row = 0
        # Spacing tuned to target density (~r=0.1014)
        dx = 0.205
        dy = 0.176
        while idx < n:
            shift = (row % 2) * (dx / 2)
            x = 0.085 + shift
            while x + 0.085 <= 1.0 and idx < n:
                v[3*idx] = x
                v[3*idx+1] = y
                idx += 1
                x += dx
            y += dy
            row += 1
    else:
        # Random valid start
        v[0::3] = np.random.uniform(0.2, 0.8, n)
        v[1::3] = np.random.uniform(0.2, 0.8, n)
        
    return v

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    best_sum = -1.0
    best_vars = None
    
    # Multiple restarts from hexagonal and random configurations
    methods = ['hex'] + ['rand'] * 8
    for m in methods:
        x0 = get_initial_config(m)
        try:
            res = minimize(compute_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 10000, 'ftol': 1e-14})
            
            r_opt = res.x[2::3]
            cx_opt = res.x[0::3]
            cy_opt = res.x[1::3]
            
            # Quick validity check to filter out clearly invalid results
            valid = True
            if np.any(r_opt < 0): 
                valid = False
            elif np.any(cx_opt - r_opt < -1e-6) or np.any(cx_opt + r_opt > 1.0 + 1e-6): 
                valid = False
            elif np.any(cy_opt - r_opt < -1e-6) or np.any(cy_opt + r_opt > 1.0 + 1e-6): 
                valid = False
            
            if valid:
                dist = np.sqrt((cx_opt[:, None]-cx_opt[None, :])**2 + (cy_opt[:, None]-cy_opt[None, :])**2)
                r_sum = r_opt[:, None] + r_opt[None, :]
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.any(dist[mask] < r_sum[mask] - 1e-6):
                    valid = False
                
            if valid:
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_vars = res.x
        except Exception:
            continue
            
    if best_vars is None:
        best_vars = get_initial_config('hex')
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Final safety projection to guarantee strict validity within tolerance
    # Calculate max scaling factor k such that k*radii satisfies all constraints
    k = 1.0
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if r < 1e-12: 
            continue
        k = min(k, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i]-centers[j])
            rs = radii[i] + radii[j]
            if rs < 1e-12: 
                continue
            k = min(k, d/rs)
            
    # Apply shrink with minimal margin for numerical safety
    radii *= max(k * 0.999999, 0.0)
    
    return centers, radii, np.sum(radii)
