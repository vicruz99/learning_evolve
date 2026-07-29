# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000080 (state b3333e60) state=46d4f5d3 sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def get_optimal_radii(centers):
    """Solves the LP to find radii that maximize sum(r_i) for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0-x, y, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 0.05)

def init_params_from_centers(centers, shrink=0.995):
    """Maps centers to strictly feasible (r, u, v) parameters."""
    r = get_optimal_radii(centers) * shrink
    r = np.clip(r, 1e-5, 0.49)
    denom = 1.0 - 2.0 * r
    u = (centers[:, 0] - r) / denom
    v = (centers[:, 1] - r) / denom
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    vars = np.empty(3 * N)
    vars[0::3] = r
    vars[1::3] = u
    vars[2::3] = v
    return vars

def obj_func(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[0::3])

def constr_func(vars):
    """Computes pairwise separation constraints g(vars) >= 0."""
    r = vars[0::3]
    u = vars[1::3]
    v = vars[2::3]
    # Automatic boundary satisfaction
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return dist2[mask] - rs[mask]**2

def force_init(seed):
    """Generates evenly spread initial points via force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    for _ in range(300):
        forces = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-5)
        f = 1.0 / (dists**2)
        f = np.clip(f, 0, 10.0)
        forces += np.sum(f[:, :, None] * diff / dists[:, :, None], axis=1)
        for d in range(2):
            forces[pts[:, d] < 0.1, d] += 5.0 * (0.1 - pts[pts[:, d] < 0.1, d])
            forces[pts[:, d] > 0.9, d] -= 5.0 * (pts[pts[:, d] > 0.9, d] - 0.9)
        forces -= 2.0 * (pts - 0.5)
        pts += 0.005 * forces
        pts = np.clip(pts, 0.01, 0.99)
    return pts

def generate_hex_centers(scale=1.0, rot=0.0, seed=0):
    """Generates hexagonal lattice points with rotation and jitter."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.095 * scale
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
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        mat = np.array([[c, -s], [s, c]])
        pts = pts - 0.5
        pts = pts @ mat.T
        pts += 0.5
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(1e-5, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constr_func}
    
    best_sum = -np.inf
    best_v = None
    
    configs = []
    
    # 1. Hexagonal variations
    for s in range(15):
        scale = 0.9 + 0.1 * (s % 3)
        rot = -0.2 + 0.2 * (s // 3)
        configs.append(generate_hex_centers(scale, rot, seed=s))
        
    # 2. Force directed
    for s in range(10):
        configs.append(force_init(seed=s))
        
    # 3. Grid + center
    grid_pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    grid_pts = np.vstack([grid_pts, [0.5, 0.5]])
    configs.append(grid_pts)
    
    # Phase 1: Multi-start SLSQP
    for idx, centers in enumerate(configs):
        v0 = init_params_from_centers(centers)
        try:
            res = minimize(obj_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if np.min(constr_func(res.x)) >= -1e-7:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        for _ in range(20):
            v_pert = best_v.copy()
            v_pert[0::3] += np.random.uniform(-0.003, 0.003, N)
            v_pert[1::3] += np.random.uniform(-0.005, 0.005, N)
            v_pert[2::3] += np.random.uniform(-0.005, 0.005, N)
            v_pert[0::3] = np.clip(v_pert[0::3], 1e-5, 0.49)
            v_pert[1::3] = np.clip(v_pert[1::3], 0.0, 1.0)
            v_pert[2::3] = np.clip(v_pert[2::3], 0.0, 1.0)
            
            try:
                res = minimize(obj_func, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 2000, 'ftol': 1e-13, 'disp': False})
                if np.min(constr_func(res.x)) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_v = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision final polish
        try:
            res_final = minimize(obj_func, best_v, method='SLSQP', bounds=bounds, constraints=cons,
                                 options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if np.min(constr_func(res_final.x)) >= -1e-8:
                best_v = res_final.x
                best_sum = -res_final.fun
        except Exception:
            pass

    # Fallback safety net
    if best_v is None:
        centers_f = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        centers_f = np.vstack([centers_f, [0.5, 0.5]])
        best_v = init_params_from_centers(centers_f, shrink=0.9)
        best_sum = np.sum(best_v[0::3])
        
    # Reconstruct physical centers from optimized parameters
    r_opt = best_v[0::3]
    u_opt = best_v[1::3]
    v_opt = best_v[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
