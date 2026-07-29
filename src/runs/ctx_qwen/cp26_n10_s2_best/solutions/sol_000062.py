# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=fdf217d3 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraint_func(v, n):
    """Vectorized inequality constraints: boundaries and pairwise non-overlap."""
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    c_list = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_list.append(centers[:, 0] - radii)
    c_list.append(1.0 - centers[:, 0] - radii)
    c_list.append(centers[:, 1] - radii)
    c_list.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap: dist >= r_i + r_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c_list.append((dists - r_sum)[mask])
    
    return np.concatenate(c_list)

def get_hex_init(n, seed):
    """Generate a hexagonal lattice initialization with random perturbation."""
    np.random.seed(seed)
    r_est = 0.09
    centers = []
    y = r_est
    row = 0
    while len(centers) < n + 10:
        x_start = r_est + (row % 2) * r_est
        x = x_start
        while x <= 1.0 - r_est:
            centers.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3) * r_est
        row += 1
        
    centers = np.array(centers[:n])
    # Add jitter to break symmetry and explore space
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Start with small radii to guarantee initial feasibility
    r_init = np.full(n, 0.03)
    return np.concatenate([centers.flatten(), r_init])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    best_sum = 0.0
    best_vars = None
    
    # Phase 1: Cold starts from diverse hexagonal perturbations
    for seed in range(20):
        x0 = get_hex_init(n, seed)
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, args=(n,),
                           options={'maxiter': 10000, 'ftol': 1e-12})
            if -res.fun > best_sum:
                cons_vals = constraint_func(res.x, n)
                # Accept if mostly feasible (allow tiny numerical slack during opt)
                if np.min(cons_vals) >= -1e-7:
                    best_sum = -res.fun
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Warm starts to escape local minima from the best found solution
    if best_vars is not None:
        for seed in range(10):
            np.random.seed(seed + 100)
            x0 = best_vars.copy()
            # Small random perturbations
            x0[:2*n] += np.random.uniform(-0.005, 0.005, size=2*n)
            x0[2*n:] += np.random.uniform(-0.002, 0.002, size=n)
            # Enforce bounds explicitly before optimization
            x0 = np.clip(x0, np.zeros(3*n), np.concatenate([np.ones(2*n), 0.5*np.ones(n)]))
            
            try:
                res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                               constraints=constraints, args=(n,),
                               options={'maxiter': 8000, 'ftol': 1e-12})
                if -res.fun > best_sum:
                    cons_vals = constraint_func(res.x, n)
                    if np.min(cons_vals) >= -1e-7:
                        best_sum = -res.fun
                        best_vars = res.x.copy()
            except Exception:
                continue
                
    # Fallback if optimization completely fails
    if best_vars is None:
        centers = np.tile([0.5, 0.5], (n, 1))
        radii = np.zeros(n)
        return centers, radii, 0.0

    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # Clip radii to stay within boundaries strictly
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0],
                       centers[i, 1], 1.0 - centers[i, 1])
        
    # Iteratively resolve pairwise overlaps with a safety margin
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    excess = (radii[i] + radii[j] - dist) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - excess)
                    radii[j] = max(0.0, radii[j] - excess)
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
