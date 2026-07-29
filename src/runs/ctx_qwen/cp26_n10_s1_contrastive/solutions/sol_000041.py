# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=95ebc3f4 sum of radii=2.622763 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
# Precompute indices for upper triangular part to speed up constraint evaluation
TRI_INDICES = np.triu_indices(N_CIRCLES, k=1)

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[:N_CIRCLES])

def constraints(x):
    """
    Constraint function: ensures pairwise non-overlap.
    Boundary constraints are satisfied automatically by the variable transformation.
    Returns array of constraint values >= 0.
    """
    r = x[:N_CIRCLES]
    u = x[N_CIRCLES:2*N_CIRCLES]
    v = x[2*N_CIRCLES:3*N_CIRCLES]
    
    # Transformation ensures r <= x_pos <= 1-r and r <= y_pos <= 1-r
    scale = 1.0 - 2.0 * r
    x_pos = r + u * scale
    y_pos = r + v * scale
    
    # Pairwise squared distances
    dx = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
    dy = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    # Minimum allowed squared distances
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    d2_min = r_sum**2
    
    # Extract upper triangular constraints (i < j)
    return d2[TRI_INDICES] - d2_min[TRI_INDICES]

def run_packing():
    best_val = -np.inf
    best_sol = None
    
    # Bounds: r in [1e-5, 0.49], u in [0, 1], v in [0, 1]
    bounds = [(1e-5, 0.49)] * N_CIRCLES + [(0.0, 1.0)] * N_CIRCLES + [(0.0, 1.0)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    inits = []
    
    # 1. Hexagonal lattice initialization
    r0 = 0.085
    pts = []
    y = r0
    row = 0
    while len(pts) < N_CIRCLES:
        x = r0
        while x <= 1.0 - r0 and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    r_vec = np.full(N_CIRCLES, r0)
    denom = 1.0 - 2.0 * r0
    u_vec = np.array([(p[0] - r0) / denom for p in pts])
    v_vec = np.array([(p[1] - r0) / denom for p in pts])
    inits.append(np.concatenate([r_vec, u_vec, v_vec]))
    
    # 2. Grid initialization
    r0 = 0.09
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    pts.append([0.5, 0.5])  # 26th circle
    pts = pts[:N_CIRCLES]
    denom = 1.0 - 2.0 * r0
    u_vec = np.array([(p[0] - r0) / denom for p in pts])
    v_vec = np.array([(p[1] - r0) / denom for p in pts])
    inits.append(np.concatenate([np.full(N_CIRCLES, r0), u_vec, v_vec]))
    
    # 3. Random feasible-ish starts
    for seed in range(15):
        np.random.seed(seed)
        r_rand = np.random.uniform(0.06, 0.11, N_CIRCLES)
        u_rand = np.random.uniform(0, 1, N_CIRCLES)
        v_rand = np.random.uniform(0, 1, N_CIRCLES)
        inits.append(np.concatenate([r_rand, u_rand, v_rand]))
        
    # 4. Perturbed hexagonal starts
    for seed in range(10):
        np.random.seed(seed + 1000)
        pts_p = [list(p) for p in pts]
        r_pert = np.full(N_CIRCLES, 0.09) + np.random.uniform(-0.01, 0.01, N_CIRCLES)
        for k in range(N_CIRCLES):
            pts_p[k][0] += np.random.uniform(-0.02, 0.02)
            pts_p[k][1] += np.random.uniform(-0.02, 0.02)
            pts_p[k][0] = np.clip(pts_p[k][0], r_pert[k], 1.0 - r_pert[k])
            pts_p[k][1] = np.clip(pts_p[k][1], r_pert[k], 1.0 - r_pert[k])
        denom_p = 1.0 - 2.0 * r_pert
        u_p = np.array([(pts_p[k][0] - r_pert[k]) / denom_p[k] for k in range(N_CIRCLES)])
        v_p = np.array([(pts_p[k][1] - r_pert[k]) / denom_p[k] for k in range(N_CIRCLES)])
        inits.append(np.concatenate([r_pert, u_p, v_p]))
        
    # Run optimization from each start
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            val = -res.fun
            # Verify strict feasibility (allow tiny numerical tolerance)
            if np.min(constraints(res.x)) >= -1e-7:
                if val > best_val:
                    best_val = val
                    best_sol = res.x.copy()
        except Exception:
            continue
            
    # Final high-precision polishing
    if best_sol is not None:
        try:
            res2 = minimize(objective, best_sol, method='SLSQP', bounds=bounds,
                            constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
            if np.min(constraints(res2.x)) >= -1e-7:
                best_sol = res2.x
        except Exception:
            pass
            
    # Fallback if optimization fails completely
    if best_sol is None:
        r_f = 0.05
        x_f = np.tile(np.linspace(0.1, 0.9, 5), 5)
        y_f = np.repeat(np.linspace(0.1, 0.9, 5), 5)
        centers = np.column_stack((x_f, y_f))
        return centers, np.full(N_CIRCLES, r_f), r_f * N_CIRCLES
        
    # Reconstruct physical coordinates
    r_opt = best_sol[:N_CIRCLES]
    u_opt = best_sol[N_CIRCLES:2*N_CIRCLES]
    v_opt = best_sol[2*N_CIRCLES:3*N_CIRCLES]
    scale = 1.0 - 2.0 * r_opt
    x_opt = r_opt + u_opt * scale
    y_opt = r_opt + v_opt * scale
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, np.sum(r_opt)
