# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e9cb3956) state=59a8168f sum of radii=2.617507 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars, n):
    """Compute boundary and non-overlap constraints for SLSQP.
    Returns an array where all elements must be >= 0."""
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    c = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c.append(x - r)
    c.append(1 - x - r)
    c.append(y - r)
    c.append(1 - y - r)
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    # Only need upper triangle to avoid redundancy
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx*dx + dy*dy)
    tri_i, tri_j = np.triu_indices(n, k=1)
    c.append(dists[tri_i, tri_j] - r[tri_i] - r[tri_j])
    
    return np.concatenate(c)

def objective(vars, n):
    """Negative sum of radii (since minimize finds minima)."""
    return -np.sum(vars[2*n:])

def run_packing():
    n = 26
    best_res = None
    best_sum = -1.0
    
    for seed in range(3):
        np.random.seed(seed)
        # Feasible initial guess: 5x5 grid + 1 center circle
        pts = np.array([[0.2 + i*0.2, 0.2 + j*0.2] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [[0.5, 0.5]]])
        # Add small perturbation to break symmetry and aid optimization
        pts += np.random.uniform(-0.01, 0.01, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        x0 = pts[:, 0]
        y0 = pts[:, 1]
        r0 = np.full(n, 0.09)
        vars0 = np.concatenate([x0, y0, r0])
        
        bounds = [(0, 1)]*n + [(0, 1)]*n + [(0, 0.5)]*n
        cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
        
        try:
            res = minimize(objective, vars0, args=(n,), method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            
            x_opt = res.x[:n]
            y_opt = res.x[n:2*n]
            r_opt = res.x[2*n:]
            
            # Post-process: strictly enforce boundary constraints
            r_opt = np.minimum(r_opt, np.minimum(x_opt, 1-x_opt))
            r_opt = np.minimum(r_opt, np.minimum(y_opt, 1-y_opt))
            
            # Post-process: strictly enforce non-overlap constraints
            for _ in range(50):
                changed = False
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-12:
                            scale = d / (r_opt[i] + r_opt[j])
                            # Reduce radii geometrically to maintain balance
                            r_opt[i] *= np.sqrt(scale)
                            r_opt[j] *= np.sqrt(scale)
                            changed = True
                if not changed:
                    break
            
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_res = (np.vstack([x_opt, y_opt]).T, r_opt, current_sum)
        except Exception:
            continue
            
    if best_res is None:
        # Fallback configuration (grid + center)
        x = np.tile(np.linspace(0.2, 0.8, 5), 5).reshape(-1, 1)
        y = np.repeat(np.linspace(0.2, 0.8, 5), 5).reshape(-1, 1)
        x = np.vstack([x, [[0.5]]])
        y = np.vstack([y, [[0.5]]])
        r = np.full(26, 0.09)
        return np.hstack([x, y]), r, 2.34
        
    return best_res
