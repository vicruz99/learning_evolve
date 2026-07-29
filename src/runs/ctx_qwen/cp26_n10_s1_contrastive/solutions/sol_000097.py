# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000064 (state 39c4bccd) state=77c3af13 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_radii_lp(centers):
    """Solves LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
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
        
    return np.full(n, 1e-5), 2.6e-4

def joint_objective(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2::3])

def joint_constraints(v):
    """Inequality constraints: g(v) >= 0."""
    cx, cy, r = v[0::3], v[1::3], v[2::3]
    c = []
    # Boundary containment
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def generate_config(seed, rot, scale):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    np.random.seed(seed)
    pts = []
    r_est = 0.09
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
    pts += np.random.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def run_packing():
    np.random.seed(42)
    best_sum = -np.inf
    best_v = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': joint_constraints}
    
    # Phase 1: Diverse global search
    configs = []
    for s in range(40):
        rot = np.random.uniform(-0.25, 0.25)
        scale = np.random.uniform(0.85, 1.15)
        configs.append((s, rot, scale))
        
    for s, rot, scale in configs:
        centers_init = generate_config(s, rot, scale)
        r_init, _ = solve_radii_lp(centers_init)
        # Scale down slightly to ensure strict interior feasibility for SLSQP
        r_init = np.maximum(r_init * 0.995, 1e-6)
        
        v0 = np.zeros(3 * N)
        v0[0::3] = centers_init[:, 0]
        v0[1::3] = centers_init[:, 1]
        v0[2::3] = r_init
        
        try:
            res = minimize(joint_objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = joint_constraints(res.x)
                if np.min(c_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement
    if best_v is not None:
        for _ in range(25):
            v_p = best_v.copy()
            v_p += np.random.randn(3 * N) * 0.003
            v_p[0::3] = np.clip(v_p[0::3], 0.01, 0.99)
            v_p[1::3] = np.clip(v_p[1::3], 0.01, 0.99)
            v_p[2::3] = np.clip(v_p[2::3], 1e-6, 0.49)
            
            try:
                res_p = minimize(joint_objective, v_p, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
                if res_p.success:
                    if np.min(joint_constraints(res_p.x)) >= -1e-8:
                        s = -res_p.fun
                        if s > best_sum:
                            best_sum = s
                            best_v = res_p.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res_f = minimize(joint_objective, best_v, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res_f.success and np.min(joint_constraints(res_f.x)) >= -1e-9:
                best_v = res_f.x
                best_sum = -res_f.fun
        except Exception:
            pass
            
    # Fallback (should rarely be reached)
    if best_v is None:
        centers = generate_config(0, 0.0, 1.0)
        radii, _ = solve_radii_lp(centers)
        best_v = np.zeros(3 * N)
        best_v[0::3] = centers[:, 0]
        best_v[1::3] = centers[:, 1]
        best_v[2::3] = np.maximum(radii, 1e-6)
        best_sum = np.sum(best_v[2::3])
        
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3]
    return centers, radii, float(best_sum)
