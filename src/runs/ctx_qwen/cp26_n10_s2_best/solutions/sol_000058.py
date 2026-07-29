# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=d534c55b sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraint_boundaries(v, n):
    """Ensure circles stay within [0,1]x[0,1]."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    c = np.empty(4*n)
    c[0::4] = x - r
    c[1::4] = 1.0 - x - r
    c[2::4] = y - r
    c[3::4] = 1.0 - y - r
    return c

def constraint_pairs(v, n, i_idx, j_idx):
    """Ensure non-overlap using squared distances for smoothness."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dr = r[i_idx] + r[j_idx]
    return dx**2 + dy**2 - dr**2

def run_packing():
    n = 26
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Pre-configure constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_boundaries, 'args': (n,)},
        {'type': 'ineq', 'fun': constraint_pairs, 'args': (n, i_idx, j_idx)}
    ]
    
    best_sum = 0.0
    best_v = None
    
    # Phase 1: Broad exploration with hexagonal lattice starts
    for seed in range(12):
        np.random.seed(seed)
        
        # Hexagonal lattice generation
        r_init = 0.06
        centers = []
        y_pos = r_init
        row = 0
        while len(centers) < n + 10:
            x_start = r_init + (row % 2) * r_init
            x_pos = x_start
            while x_pos <= 1.0 - r_init:
                centers.append([x_pos, y_pos])
                x_pos += 2.0 * r_init
            y_pos += np.sqrt(3.0) * r_init
            row += 1
            
        centers = np.array(centers[:n])
        
        # Break symmetry with jitter
        centers += np.random.uniform(-0.015, 0.015, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Shuffle order to prevent index-bias in optimizer
        perm = np.random.permutation(n)
        centers = centers[perm]
        
        v0 = np.concatenate([centers[:, 0], centers[:, 1], np.full(n, 0.05)])
        
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            
            if np.isfinite(res.fun):
                current_sum = -res.fun
                if current_sum > best_sum:
                    # Verify feasibility
                    b = constraint_boundaries(res.x, n)
                    p = constraint_pairs(res.x, n, i_idx, j_idx)
                    if np.all(b >= -1e-5) and np.all(p >= -1e-5):
                        best_sum = current_sum
                        best_v = res.x.copy()
        except Exception:
            continue

    # Phase 2: Refinement around best solution to escape local optima
    if best_v is not None:
        for ref_seed in range(5):
            np.random.seed(100 + ref_seed)
            v_pert = best_v.copy()
            # Perturb centers more than radii
            v_pert[:2*n] += np.random.uniform(-0.005, 0.005, 2*n)
            v_pert[2*n:] += np.random.uniform(-0.002, 0.002, n)
            v_pert = np.clip(v_pert, 0.01, 0.99) # Keep strictly inside
            
            try:
                res = minimize(objective_func, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                
                if np.isfinite(res.fun):
                    current_sum = -res.fun
                    if current_sum > best_sum:
                        b = constraint_boundaries(res.x, n)
                        p = constraint_pairs(res.x, n, i_idx, j_idx)
                        if np.all(b >= -1e-5) and np.all(p >= -1e-5):
                            best_sum = current_sum
                            best_v = res.x.copy()
            except Exception:
                continue

    if best_v is None:
        # Fallback valid configuration
        centers = np.array([[0.5, 0.5]] * n)
        radii = np.zeros(n)
        return centers, radii, 0.0

    x_sol = best_v[:n]
    y_sol = best_v[n:2*n]
    r_sol = best_v[2*n:]
    
    # Strict post-processing to guarantee validator tolerance
    for _ in range(10):
        changed = False
        for i in range(n):
            # Boundary limits
            r_lim = min(r_sol[i], x_sol[i], 1.0-x_sol[i], y_sol[i], 1.0-y_sol[i])
            if r_lim < r_sol[i] - 1e-9:
                r_sol[i] = r_lim
                changed = True
                
            # Pairwise limits
            for j in range(i+1, n):
                dist = np.hypot(x_sol[i]-x_sol[j], y_sol[i]-y_sol[j])
                sum_r = r_sol[i] + r_sol[j]
                if sum_r > dist:
                    shrink = (sum_r - dist) / 2.0 + 1e-9
                    r_sol[i] = max(0.0, r_sol[i] - shrink)
                    r_sol[j] = max(0.0, r_sol[j] - shrink)
                    changed = True
        if not changed:
            break
            
    final_centers = np.column_stack([x_sol, y_sol])
    return final_centers, r_sol, float(np.sum(r_sol))
