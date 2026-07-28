# sol_000116 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=9770a7aa sum of radii=2.505333 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def get_initial_t(cx, cy):
    """Compute a feasible initial t for the max-min optimization."""
    b = np.minimum(np.minimum(cx, 1.0 - cx), np.minimum(cy, 1.0 - cy))
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    return min(np.min(b), np.min(d) / 2.0)

def objective_maxmin(vars):
    """Minimize -t => maximize t."""
    return -vars[-1]

def constraints_maxmin(vars):
    """Inequality constraints for equal-radius packing: all clearances >= 2t."""
    n = N
    t = vars[-1]
    cx = vars[:n]
    cy = vars[n:2*n]
    
    # Boundary constraints: x >= t, 1-x >= t, y >= t, 1-y >= t
    b = np.concatenate([cx - t, 1.0 - cx - t, cy - t, 1.0 - cy - t])
    
    # Pairwise constraints: dist(i,j) >= 2t
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    triu = np.triu_indices(n, k=1)
    p = d[triu] - 2.0 * t
    
    return np.concatenate([b, p])

def solve_lp_radii(centers):
    """Maximize sum of radii for fixed centers using Linear Programming."""
    n = centers.shape[0]
    # Maximum radius allowed by boundaries
    lim = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                     np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lim = np.maximum(lim, 0.0)
    
    c_obj = -np.ones(n)
    bounds = [(0.0, lim[i]) for i in range(n)]
    
    # Pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    triu = np.triu_indices(n, k=1)
    A_ub = np.zeros((n * (n - 1) // 2, n))
    b_ub = np.zeros(n * (n - 1) // 2)
    
    idx = 0
    for i, j in zip(triu[0], triu[1]):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dists[i, j]
        idx += 1
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    starts = []
    for seed in range(8):
        pts = []
        y = 0.1
        row = 0
        np.random.seed(seed)
        # Slightly vary hex spacing to explore different lattice alignments
        dx_sp = 0.09 + np.random.uniform(-0.01, 0.01)
        dy_sp = dx_sp * np.sqrt(3) / 2
        
        while y < 0.95:
            x = 0.1
            shift = dx_sp / 2 if row % 2 == 1 else 0.0
            while x < 0.95:
                pts.append([x + shift, y])
                x += dx_sp
            y += dy_sp
            row += 1
            
        pts = np.array(pts)
        # Pick N points closest to the center for a compact, symmetric start
        dists = np.sum((pts - 0.5) ** 2, axis=1)
        idx = np.argsort(dists)[:N]
        starts.append(pts[idx])
        
    # Add a structured grid fallback
    grid = []
    for i in range(5):
        for j in range(5):
            grid.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    grid.append([0.5, 0.5])
    starts.append(np.array(grid[:N]))
    
    cons_maxmin = {'type': 'ineq', 'fun': constraints_maxmin}
    
    # Phase 1: Optimize centers to maximize minimum clearance
    for cfg in starts:
        cx, cy = cfg[:, 0], cfg[:, 1]
        t0 = get_initial_t(cx, cy) * 0.95
        if t0 < 0.02:
            t0 = 0.02
        
        x0 = np.concatenate([cx, cy, [t0]])
        bounds = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.15)]
        
        try:
            res = minimize(
                objective_maxmin, x0, method='SLSQP', bounds=bounds, 
                constraints=cons_maxmin, options={'maxiter': 5000, 'ftol': 1e-12}
            )
            
            cx_opt = res.x[:N]
            cy_opt = res.x[N:2*N]
            centers_opt = np.column_stack((cx_opt, cy_opt))
            
            # Phase 2: Solve LP to extract optimal radii for these centers
            radii, s = solve_lp_radii(centers_opt)
            if radii is not None and s > best_sum:
                best_sum = s
                best_centers = centers_opt
                best_radii = radii
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = starts[0]
        best_radii = np.full(N, 0.09)
        best_sum = np.sum(best_radii)
        
    # Apply tiny safety margin to guarantee strict validity against 1e-12 tolerance
    best_radii *= 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
