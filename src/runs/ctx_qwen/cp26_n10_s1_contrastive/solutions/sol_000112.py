# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000104 (state be0dee1d) state=111b2725 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def obj_func(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[0::3])

def constr_func(params):
    """Pairwise non-overlap constraints: dist_sq >= (r_i + r_j)^2."""
    r = params[0::3]
    u = params[1::3]
    v = params[2::3]
    
    # Parameterization guarantees boundary containment automatically
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dist2 = dx**2 + dy**2
    
    rs = r[I_IDX] + r[J_IDX]
    return dist2 - rs**2

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.full(n, 1e-5), 0.0

def force_init(seed):
    """Generates a strictly feasible initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    r_curr = np.full(N, 0.08)
    
    for _ in range(500):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-6)
        rep = 1.0 / dists**2
        np.fill_diagonal(rep, 0.0)
        forces = np.sum(rep[:, :, None] * diff / dists[:, :, None], axis=1)
        
        # Wall repulsion
        for d in range(2):
            forces[:, d] += 10.0 * np.maximum(0, r_curr - pts[:, d])
            forces[:, d] -= 10.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
            
        pts += 0.003 * forces
        pts = np.clip(pts, 0.001, 0.999)
        
        for i in range(N):
            dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            dp = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            r_curr[i] = 0.95 * min(dw, dp / 2.0)
            
    r_final = np.zeros(N)
    for i in range(N):
        dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        dists_i = np.linalg.norm(pts[i] - pts, axis=1)
        dists_i[i] = np.inf
        dp = np.min(dists_i)
        r_final[i] = 0.98 * min(dw, dp / 2.0)
        
    denom = np.clip(1.0 - 2.0 * r_final, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r_final) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r_final) / denom, 0.0, 1.0)
    
    params = np.empty(3 * N)
    params[0::3] = r_final
    params[1::3] = u
    params[2::3] = v
    return params

def hex_init(seed, rot=0.0, scale=1.0):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    while len(pts) < N:
        shift = (row % 2) * r_est
        x = r_est + shift
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
        
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    r_vals = np.zeros(N)
    for i in range(N):
        dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        dists_i = np.linalg.norm(pts[i] - pts, axis=1)
        dists_i[i] = np.inf
        dp = np.min(dists_i)
        r_vals[i] = 0.9 * min(dw, dp / 2.0)
        
    denom = np.clip(1.0 - 2.0 * r_vals, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r_vals) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r_vals) / denom, 0.0, 1.0)
    
    params = np.empty(3 * N)
    params[0::3] = r_vals
    params[1::3] = u
    params[2::3] = v
    return params

def run_packing():
    np.random.seed(42)
    bounds = [(1e-5, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constr_func}
    
    best_params = None
    best_sum = -np.inf
    
    # Generate diverse initializations
    inits = []
    for s in range(25):
        inits.append(force_init(s))
        
    for s in range(15):
        rot = np.random.uniform(-0.2, 0.2)
        sc = np.random.uniform(0.9, 1.1)
        inits.append(hex_init(s, rot, sc))
        
    for s in range(10):
        pts = np.random.rand(N, 2)
        pts = pts[pts[:, 1].argsort()]
        for i in range(N):
            pts[i, 0] = 0.05 + pts[i, 0] * 0.9 + np.random.uniform(-0.03, 0.03)
        pts = np.clip(pts, 0.02, 0.98)
        r_vals = np.zeros(N)
        for i in range(N):
            dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            dists_i = np.linalg.norm(pts[i] - pts, axis=1)
            dists_i[i] = np.inf
            dp = np.min(dists_i)
            r_vals[i] = 0.85 * min(dw, dp / 2.0)
        denom = np.clip(1.0 - 2.0 * r_vals, 1e-6, 1.0)
        u = np.clip((pts[:, 0] - r_vals) / denom, 0.0, 1.0)
        v = np.clip((pts[:, 1] - r_vals) / denom, 0.0, 1.0)
        p = np.empty(3 * N)
        p[0::3] = r_vals
        p[1::3] = u
        p[2::3] = v
        inits.append(p)

    # Phase 1: Broad search
    for p0 in inits:
        try:
            res = minimize(obj_func, p0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = constr_func(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_params = res.x.copy()
        except Exception:
            continue

    if best_params is None:
        best_params = hex_init(0)
        
    # Phase 2: Local perturbation refinement
    for _ in range(40):
        p0 = best_params + np.random.normal(0, 0.003, 3 * N)
        p0[0::3] = np.clip(p0[0::3], 1e-5, 0.5)
        p0[1::3] = np.clip(p0[1::3], 0.0, 1.0)
        p0[2::3] = np.clip(p0[2::3], 0.0, 1.0)
        try:
            res = minimize(obj_func, p0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = constr_func(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_params = res.x.copy()
        except Exception:
            continue
            
    # Phase 3: High-precision polish
    try:
        res = minimize(obj_func, best_params, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if res.success and np.min(constr_func(res.x)) >= -1e-9:
            best_params = res.x
            best_sum = -res.fun
    except Exception:
        pass
        
    # Reconstruct centers
    r_opt = best_params[0::3]
    u_opt = best_params[1::3]
    v_opt = best_params[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    # Phase 4: Exact LP refinement for radii
    radii_lp, sum_lp = solve_lp_radii(centers)
    if sum_lp > best_sum:
        best_sum = sum_lp
        radii = radii_lp
    else:
        radii = r_opt
        
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(best_sum)
