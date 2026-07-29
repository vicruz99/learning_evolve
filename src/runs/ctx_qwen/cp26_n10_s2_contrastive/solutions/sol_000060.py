# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000038 (state cf517c54) state=5df37737 sum of radii=2.613917 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_optimal_radii(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(radii)
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    # Upper bounds for radii from boundaries: r_i <= min(x, 1-x, y, 1-y)
    ub = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    ub = np.maximum(ub, 0.0)
    bounds = [(0.0, u) for u in ub]
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    c_obj = -np.ones(n)
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def objective(x):
        """Minimize negative sum of radii."""
        return -np.sum(x[2 * n:])
        
    def constraints(x):
        """Inequality constraints: boundary and pairwise non-overlap."""
        cx = x[:n]
        cy = x[n : 2 * n]
        r = x[2 * n :]
        
        c = np.empty(4 * n + n * (n - 1) // 2)
        c[:n] = cx - r
        c[n : 2 * n] = 1.0 - cx - r
        c[2 * n : 3 * n] = cy - r
        c[3 * n : 4 * n] = 1.0 - cy - r
        
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dists = np.hypot(dx, dy)
        r_sum = r[:, None] + r[None, :]
        
        idx = np.triu_indices(n, k=1)
        c[4 * n :] = dists[idx] - r_sum[idx]
        return c

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_x = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with varying spacing
    for spacing in [0.18, 0.19, 0.20, 0.21, 0.22]:
        pts = []
        y = spacing / 2.0
        row = 0
        while len(pts) < n:
            x = spacing / 2.0 if row % 2 == 0 else spacing
            while x < 1.0 - spacing / 2.0 and len(pts) < n:
                pts.append([x, y])
                x += spacing
            y += spacing * np.sqrt(3.0) / 2.0
            row += 1
        inits.append(np.concatenate([np.array(pts[:n]).flatten(), np.full(n, 0.07)]))
        
    # 2. Perturbed hex grids
    for seed in range(30):
        rng = np.random.RandomState(seed)
        base = inits[0].copy().reshape(-1, 2)
        p = base + rng.normal(0, 0.025, base.shape)
        p = np.clip(p, 0.05, 0.95)
        inits.append(np.concatenate([p.flatten(), np.full(n, 0.065)]))
        
    # 3. Random placements (small radii to ensure initial feasibility)
    for seed in range(20):
        rng = np.random.RandomState(seed)
        cx = rng.uniform(0.15, 0.85, n)
        cy = rng.uniform(0.15, 0.85, n)
        inits.append(np.concatenate([cx, cy, np.full(n, 0.04)]))
        
    # Main optimization loop
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Local refinement around the best solution to escape shallow minima
    if best_x is not None:
        for trial in range(20):
            rng = np.random.RandomState(trial + 100)
            x_pert = best_x + rng.normal(0, 0.004, len(best_x))
            x_pert = np.clip(x_pert, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
                if res.success and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
            except Exception:
                continue
                
    if best_x is None:
        best_x = inits[0]
        
    cx = best_x[:n]
    cy = best_x[n : 2 * n]
    r = best_x[2 * n :]
    
    centers = np.column_stack((cx, cy))
    
    # Final polish: optimize radii for fixed centers using LP.
    # This exactly maximizes sum(r) for the found topology, removing numerical slack.
    _, polished_r = get_optimal_radii(centers)
    
    # Strict validity enforcement against floating point drift
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        polished_r[i] = min(polished_r[i], mx)
    polished_r = np.maximum(polished_r, 0.0)
    
    final_sum = np.sum(polished_r)
    return centers, polished_r, float(final_sum)
