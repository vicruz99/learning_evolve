# sol_000335 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 29661f66) state=5a4feae0 sum of radii=2.612478 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_config(n, pattern='hex', seed=0):
    """Generate a valid initial configuration for optimization."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08)
    
    if pattern == 'hex':
        # Staggered hexagonal arrangement approximates optimal density
        row_counts = [6, 5, 6, 5, 4]
        idx = 0
        for k, count in enumerate(row_counts):
            y = (k + 0.5) / 5.0
            for j in range(count):
                x = (j + 0.5) / count
                centers[idx] = [x, y]
                idx += 1
    else:
        # Random valid packing via rejection sampling
        r = 0.06
        placed = []
        attempts = 0
        while len(placed) < n and attempts < 20000:
            attempts += 1
            cx, cy = np.random.rand(2)
            valid = True
            if cx - r < 0 or cx + r > 1 or cy - r < 0 or cy + r > 1:
                valid = False
            else:
                for px, py in placed:
                    if np.hypot(cx - px, cy - py) < 2 * r:
                        valid = False
                        break
            if valid:
                placed.append((cx, cy))
        if len(placed) == n:
            centers[:n] = placed
            radii[:] = r
        else:
            return get_initial_config(n, 'hex', seed)
            
    # Add small perturbation to break exact symmetries
    noise = np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers + noise, 0.05, 0.95)
    return centers, radii

def constraint_func(v, n=26):
    """Evaluate all packing constraints. Returns array of constraint values (must be >= 0)."""
    x = v[:2*n].reshape((n, 2))
    r = v[2*n:]
    c = []
    # Boundary constraints
    for i in range(n):
        c.append(x[i, 0] - r[i])          # x - r >= 0
        c.append(1 - x[i, 0] - r[i])      # 1 - x - r >= 0
        c.append(x[i, 1] - r[i])          # y - r >= 0
        c.append(1 - x[i, 1] - r[i])      # 1 - y - r >= 0
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            c.append(np.sqrt(dx * dx + dy * dy) - r[i] - r[j])
    return np.array(c)

def objective_func(v, n=26):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(v[2*n:])

def run_packing():
    """Run optimization and return best valid packing."""
    n = 26
    best_sum = -1.0
    best_x = None
    
    # Try multiple initial configurations to avoid local minima
    configs = [
        ('hex', 0), ('hex', 1), ('hex', 2),
        ('random', 0), ('random', 1)
    ]
    
    for pat, seed in configs:
        centers, radii = get_initial_config(n, pat, seed)
        x0 = np.concatenate([centers.ravel(), radii])
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12})
            s = np.sum(res.x[2*n:])
            if s > best_sum:
                best_sum = s
                best_x = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails entirely
    if best_x is None:
        centers, radii = get_initial_config(n, 'hex', 0)
        return centers, radii, np.sum(radii)
        
    centers_opt = best_x[:2*n].reshape((n, 2))
    radii_opt = best_x[2*n:]
    
    # Post-processing: ensure strict validity and clip numerical drift
    radii_opt = np.maximum(radii_opt, 1e-9)
    for i in range(n):
        r = radii_opt[i]
        centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1 - r)
        centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1 - r)
        
    return centers_opt, radii_opt, np.sum(radii_opt)
