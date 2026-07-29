# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state 9b0797fd) state=57c93ce5 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N_CIRCLES = 26

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Returns all inequality constraints >= 0."""
    n = N_CIRCLES
    res = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    for i in range(n):
        x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
        res.append(x - r)
        res.append(1.0 - x - r)
        res.append(y - r)
        res.append(1.0 - y - r)
        
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            dx = xi - xj
            dy = yi - yj
            res.append(dx*dx + dy*dy - (ri + rj)**2)
            
    return np.array(res)

def generate_hex_init(n, seed, r_start=0.09):
    """Generates a hexagonal lattice initialization."""
    np.random.seed(seed)
    vars = np.zeros(3 * n)
    count = 0
    y = r_start
    while count < n:
        x = r_start
        # Shift odd rows by r_start to create hex pattern
        row_idx = int(y / r_start)
        shift = r_start if (row_idx % 2 == 1) else 0.0
        
        while count < n:
            if x + r_start > 1.0:
                break
                
            cx = x + shift + np.random.uniform(-0.005, 0.005)
            cy = y + np.random.uniform(-0.005, 0.005)
            cr = r_start
            
            vars[3*count] = cx
            vars[3*count+1] = cy
            vars[3*count+2] = cr
            count += 1
            x += 2 * r_start
            
        y += math.sqrt(3) * r_start
        if y + r_start > 1.0:
            break
            
    # Fill remaining if any (rare with r=0.09)
    while count < n:
        vars[3*count] = np.random.uniform(0.15, 0.85)
        vars[3*count+1] = np.random.uniform(0.15, 0.85)
        vars[3*count+2] = 0.06
        count += 1
        
    return vars

def generate_grid_init(n, seed):
    """Generates a perturbed grid initialization."""
    np.random.seed(seed)
    vars = np.zeros(3 * n)
    
    # 5x6 grid provides 30 spots, we take first 26
    pts = []
    for r in range(6):
        for c in range(5):
            pts.append((0.08 + c * 0.18, 0.08 + r * 0.16))
            
    for i in range(n):
        if i < len(pts):
            bx, by = pts[i]
            cx = bx + np.random.uniform(-0.01, 0.01)
            cy = by + np.random.uniform(-0.01, 0.01)
            cr = 0.065
        else:
            cx = np.random.uniform(0.15, 0.85)
            cy = np.random.uniform(0.15, 0.85)
            cr = 0.05
            
        vars[3*i] = cx
        vars[3*i+1] = cy
        vars[3*i+2] = cr
        
    return vars

def ensure_feasibility(vars, n):
    """Clips variables to guarantee initial feasibility."""
    for i in range(n):
        r = vars[3*i + 2]
        if r < 0.01:
            r = 0.01
            vars[3*i + 2] = r
        vars[3*i] = np.clip(vars[3*i], r, 1.0 - r)
        vars[3*i+1] = np.clip(vars[3*i+1], r, 1.0 - r)
    return vars

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Phase 1: Multiple diverse starts
    for seed in range(20):
        if seed % 2 == 0:
            x0 = generate_hex_init(n, seed, r_start=0.09 + 0.005 * (seed % 4))
        else:
            x0 = generate_grid_init(n, seed)
            
        x0 = ensure_feasibility(x0, n)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Refinement on best solution
    if best_vars is not None:
        for _ in range(3):
            # Perturb slightly to escape local minima
            x0 = best_vars + np.random.normal(0, 1e-4, 3 * n)
            x0 = ensure_feasibility(x0, n)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_vars = res.x.copy()
            except Exception:
                break
                
    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = best_vars[3*i+2]
        
    # Final safety check: shrink radii minimally if numerical drift caused micro-overlaps
    # This ensures strict compliance with the provided validate_packing function
    valid = True
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-10:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Aggressive but minimal shrinkage to recover validity
        factor = 0.99
        for _ in range(50):
            radii *= factor
            centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
            # Quick check
            is_ok = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if np.sqrt(dx*dx+dy*dy) < radii[i] + radii[j] - 1e-12:
                        is_ok = False
                        break
                if not is_ok: break
            if is_ok:
                break
                
    return centers, radii, float(np.sum(radii))
