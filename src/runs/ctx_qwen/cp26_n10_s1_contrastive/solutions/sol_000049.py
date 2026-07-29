# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state ce356e52) state=f24b09a0 sum of radii=2.615428 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Constraint function: returns inequality constraints >= 0."""
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = rs[:, None] + rs[None, :]
    min_dist_sq = r_sum**2
    
    # Extract upper triangular part to avoid duplicates and self-pairs
    ii, jj = np.triu_indices(n, k=1)
    c = np.concatenate([c, dist_sq[ii, jj] - min_dist_sq[ii, jj]])
    
    return c

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_x = None
    
    inits = []
    
    # 1. Hexagonal initializations with various radii
    # Hexagonal packing is theoretically densest for equal circles.
    # Varying r0 explores different density regimes before optimization.
    for r0 in [0.05, 0.06, 0.07, 0.08]:
        pts = []
        dy = r0 * np.sqrt(3.0)
        y = r0
        row = 0
        while len(pts) < n:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0 and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r0
            y += dy
            row += 1
        pts = np.array(pts[:n])
        inits.append(np.column_stack((pts, np.full(n, r0))).flatten())
        
    # 2. Random perturbations of hex layout
    # Breaks symmetry and helps escape local minima caused by perfect lattice alignment.
    for seed in range(20):
        np.random.seed(seed)
        r0 = 0.065 + np.random.uniform(-0.01, 0.01)
        pts = []
        dy = r0 * np.sqrt(3.0)
        y = r0
        row = 0
        while len(pts) < n:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0 and len(pts) < n:
                pts.append([x + np.random.uniform(-0.02, 0.02), 
                            y + np.random.uniform(-0.02, 0.02)])
                x += 2.0 * r0
            y += dy
            row += 1
        pts = np.array(pts[:n])
        # Ensure initial feasibility
        pts[:, 0] = np.clip(pts[:, 0], r0, 1.0 - r0)
        pts[:, 1] = np.clip(pts[:, 1], r0, 1.0 - r0)
        inits.append(np.column_stack((pts, np.full(n, r0))).flatten())
        
    # 3. Grid initialization (alternative structural pattern)
    for r0 in [0.05, 0.06, 0.07]:
        pts = []
        step = 0.2
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i*step, 0.1 + j*step])
        pts.append([0.5, 0.5])
        pts = np.array(pts[:n])
        inits.append(np.column_stack((pts, np.full(n, r0))).flatten())

    # Run optimization from each initial guess
    for x0 in inits:
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            if res.success:
                curr_sum = -res.fun
                # Accept if constraints are satisfied (within numerical tolerance) and improves best
                if np.min(constraint_func(res.x)) >= -1e-7 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback to a valid configuration if optimization unexpectedly fails
    if best_x is None:
        r_f = 0.05
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i*0.2, 0.1 + j*0.2])
        pts.append([0.5, 0.5])
        best_x = np.column_stack((np.array(pts[:n]), np.full(n, r_f))).flatten()
        best_sum = n * r_f
        
    # Final high-precision refinement on the best found configuration
    try:
        res_final = minimize(objective_func, best_x, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
        if res_final.success and np.min(constraint_func(res_final.x)) >= -1e-7:
            best_x = res_final.x
            best_sum = -res_final.fun
    except Exception:
        pass
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    return centers, radii, float(best_sum)
