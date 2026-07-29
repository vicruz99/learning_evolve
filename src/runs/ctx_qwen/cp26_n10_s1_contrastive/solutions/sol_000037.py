# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 1103014d) state=b1379e04 sum of radii=2.622763 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def obj(vars):
    """Objective: Minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(vars[2::3])

def con(vars):
    """
    Inequality constraints: g(vars) >= 0.
    Enforces boundary containment and non-overlap.
    """
    cx = vars[0::3]
    cy = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >=0, 1-x-r >=0, y-r >=0, 1-y-r >=0
    c_bound = np.empty(4 * N)
    c_bound[0::4] = cx - r
    c_bound[1::4] = 1.0 - cx - r
    c_bound[2::4] = cy - r
    c_bound[3::4] = 1.0 - cy - r
    
    # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_pair = dist_sq[mask] - r_sum[mask]**2
    
    return np.concatenate([c_bound, c_pair])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': con}
    
    best_sum = -np.inf
    best_vars = None
    
    init_list = []
    
    # 1. Hexagonal lattice initialization
    r0 = 0.085
    pts = []
    y = r0
    row = 0
    while len(pts) < N:
        x = r0 + (row % 2) * r0
        while x <= 1.0 - r0 and len(pts) < N:
            pts.append([x, y, r0])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    init_list.append(np.array(pts).flatten())
    
    # 2. 5x5 grid + 1 in center initialization
    pts2 = []
    for i in range(5):
        for j in range(5):
            pts2.append([0.1 + i*0.2, 0.1 + j*0.2, 0.09])
    pts2.append([0.5, 0.5, 0.01])
    init_list.append(np.array(pts2).flatten())
    
    # 3. Multiple randomized feasible starts
    np.random.seed(42)
    for _ in range(15):
        cx = np.random.uniform(0.1, 0.9, N)
        cy = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.06)
        cx += np.random.uniform(-0.03, 0.03, N)
        cy += np.random.uniform(-0.03, 0.03, N)
        cx = np.clip(cx, 0.02, 0.98)
        cy = np.clip(cy, 0.02, 0.98)
        init_list.append(np.concatenate([cx, cy, r]))
        
    # Primary optimization loop
    for i, x0 in enumerate(init_list):
        # Slight perturbation to escape exact symmetries
        x0_pert = x0 + np.random.randn(len(x0)) * 0.001
        
        try:
            res = minimize(obj, x0_pert, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            if res.success:
                curr_sum = -res.fun
                # Verify constraints are satisfied within numerical tolerance
                if np.min(con(res.x)) >= -1e-7:
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    # Refinement pass on the best configuration found
    if best_vars is not None:
        try:
            res_final = minimize(obj, best_vars, method='SLSQP', bounds=bounds, 
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if res_final.success and -res_final.fun > best_sum:
                if np.min(con(res_final.x)) >= -1e-7:
                    best_vars = res_final.x
        except Exception:
            pass

        cx = best_vars[0::3]
        cy = best_vars[1::3]
        r = best_vars[2::3]
        centers = np.column_stack([cx, cy])
        return centers, r, float(np.sum(r))
    else:
        # Fallback to a known valid (though suboptimal) packing
        cx = np.linspace(0.1, 0.9, 5)
        cy = np.linspace(0.1, 0.9, 5)
        centers = np.array([(x, y) for y in cy for x in cx])
        centers = np.vstack([centers, [0.5, 0.5]])
        radii = np.full(26, 0.09)
        return centers, radii, float(np.sum(radii))
