# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=29e04639 sum of radii=2.626088 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n):
    """Compute all inequality constraints: boundaries and pairwise non-overlap."""
    xs = v[:n]
    ys = v[n:2*n]
    rs = v[2*n:]
    
    cons = []
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    cons.append(xs - rs)          # x - r >= 0
    cons.append(1.0 - xs - rs)    # 1 - x - r >= 0
    cons.append(ys - rs)          # y - r >= 0
    cons.append(1.0 - ys - rs)    # 1 - y - r >= 0
    
    # Pairwise non-overlap constraints: distance >= r_i + r_j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = rs[:, None] + rs[None, :]
    
    # Only upper triangle pairs (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append((dist - r_sum)[mask])
    
    return np.concatenate(cons)

def get_initial(n, seed, layout='hex'):
    """Generate a feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    
    if layout == 'hex':
        r = 0.085
        y = r
        row = 0
        idx = 0
        while idx < n:
            x = r if row % 2 == 0 else 2 * r
            while x <= 1 - r and idx < n:
                centers[idx] = [x, y]
                x += 2 * r
                idx += 1
            y += r * np.sqrt(3)
            row += 1
    elif layout == 'grid':
        r = 0.085
        y = r
        row = 0
        idx = 0
        while idx < n:
            x = r
            while x <= 1 - r and idx < n:
                centers[idx] = [x, y]
                x += 2 * r
                idx += 1
            y += 2 * r
            row += 1
    else:  # random
        centers = np.random.uniform(0.05, 0.95, (n, 2))
        
    # Add controlled jitter to break symmetry and avoid degenerate gradients
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Start with small radii to guarantee initial feasibility
    return np.concatenate([centers[:,0], centers[:,1], np.full(n, 0.04)])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_v = None
    best_val = -1.0
    
    # 1. Multi-start optimization with diverse layouts
    configs = []
    for s in range(10):
        for l in ['hex', 'grid', 'rand']:
            configs.append((s, l))
            
    for seed, layout in configs:
        v0 = get_initial(n, seed, layout)
        try:
            res = minimize(objective, v0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 3000, 'ftol': 1e-12})
            if -res.fun > best_val:
                # Verify feasibility before accepting
                if np.min(constraints(res.x, n)) >= -1e-7:
                    best_val = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback in case all optimizations fail (highly unlikely)
    if best_v is None:
        best_v = get_initial(n, 0, 'hex')
        best_val = np.sum(best_v[2*n:])
        
    # 2. Adaptive Refinement: Scale radii up and re-optimize to escape local minima
    current_v = best_v
    for _ in range(4):
        rs_idx = slice(2*n, 3*n)
        current_rs = current_v[rs_idx]
        if np.max(current_rs) >= 0.48:
            break
            
        # Intentionally violate constraints by scaling radii up slightly
        perturbed_v = current_v.copy()
        perturbed_v[rs_idx] *= 1.004
        
        try:
            res = minimize(objective, perturbed_v, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 2000, 'ftol': 1e-12})
            if -res.fun > best_val:
                if np.min(constraints(res.x, n)) >= -1e-7:
                    best_val = -res.fun
                    best_v = res.x.copy()
                    current_v = best_v
        except Exception:
            break
            
    # 3. Extract and strictly enforce constraints for validator compatibility
    xs = best_v[:n]
    ys = best_v[n:2*n]
    rs = best_v[2*n:]
    centers = np.column_stack((xs, ys))
    
    # Strict boundary enforcement
    margin_x = np.minimum(centers[:,0], 1.0 - centers[:,0])
    margin_y = np.minimum(centers[:,1], 1.0 - centers[:,1])
    rs = np.minimum(rs, np.minimum(margin_x, margin_y))
    rs = np.maximum(rs, 0.0)
    
    # Strict overlap enforcement with safety margin
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            if d < rs[i] + rs[j] - 1e-12:
                shrink = (rs[i] + rs[j] - d) / 2.0 + 1e-7
                rs[i] = max(0.0, rs[i] - shrink)
                rs[j] = max(0.0, rs[j] - shrink)
                
    return centers, rs, float(np.sum(rs))
