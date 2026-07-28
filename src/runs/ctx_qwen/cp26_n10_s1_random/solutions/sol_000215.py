# sol_000215 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000165 (state ab534a56) state=7c98b74c sum of radii=0.518667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26
TRIU_IDX = np.triu_indices(N_CIRCLES, k=1)

def objective_equal_r(params):
    """Objective: maximize r => minimize -r"""
    return -params[-1]

def constraints_equal_r(params):
    """Inequality constraints >= 0 for equal radius packing"""
    cx = params[:N_CIRCLES]
    cy = params[N_CIRCLES:2*N_CIRCLES]
    r = params[-1]
    
    c = []
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise squared distance >= 4r^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    c.extend(cx_m[TRIU_IDX]**2 + cy_m[TRIU_IDX]**2 - 4.0 * r**2)
    return np.array(c)

def objective_joint(params):
    """Objective: maximize sum(r) => minimize -sum(r)"""
    return -np.sum(params[2*N_CIRCLES:])

def constraints_joint(params):
    """Inequality constraints >= 0 for variable radius packing"""
    cx = params[:N_CIRCLES]
    cy = params[N_CIRCLES:2*N_CIRCLES]
    r = params[2*N_CIRCLES:]
    
    c = []
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise squared distance >= (r_i + r_j)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    c.extend(cx_m[TRIU_IDX]**2 + cy_m[TRIU_IDX]**2 - r_m[TRIU_IDX]**2)
    return np.array(c)

def solve_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers"""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m_bound = 4 * n
    m_pair = n * (n - 1) // 2
    
    A_ub = np.zeros((m_bound + m_pair, n))
    b_ub = np.zeros(m_bound + m_pair)
    bounds = []
    
    idx = 0
    for i in range(n):
        x, y = centers[i]
        mx = max(1e-9, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            A_ub[idx, i] = 1.0
            b_ub[idx] = lim
            idx += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def generate_hex_init(row_counts, r0=0.09):
    """Generates initial positions on a hexagonal lattice"""
    pts = []
    y = r0
    for i, cnt in enumerate(row_counts):
        if len(pts) >= N_CIRCLES: break
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N_CIRCLES: break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    while len(pts) < N_CIRCLES:
        pts.append([0.5, 0.5])
    pts = np.array(pts[:N_CIRCLES])
    cx_mean, cy_mean = pts.mean(axis=0)
    pts -= np.array([cx_mean - 0.5, cy_mean - 0.5])
    return np.clip(pts, 0.05, 0.95)

def run_packing():
    np.random.seed(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    row_configs = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 5, 5, 6, 4],
        [5, 6, 6, 4, 5], [5, 5, 5, 5, 6], [7, 6, 6, 7],
        [6, 7, 6, 7], [5, 7, 7, 7], [5, 6, 5, 6, 5, 1]
    ]
    
    inits = []
    for rc in row_configs:
        if sum(rc) < N_CIRCLES: continue
        base = generate_hex_init(rc)
        inits.append(base)
        for _ in range(4):
            pert = base + np.random.uniform(-0.02, 0.02, base.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    for _ in range(10):
        inits.append(np.random.uniform(0.15, 0.85, (N_CIRCLES, 2)))
        
    bounds_eq = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.05, 0.15)]
    
    # Phase 1: Equal-radius optimization
    for cfg in inits:
        x0 = np.zeros(2 * N_CIRCLES + 1)
        x0[:N_CIRCLES] = cfg[:, 0]
        x0[N_CIRCLES:2*N_CIRCLES] = cfg[:, 1]
        x0[-1] = 0.09
        
        try:
            res = minimize(objective_equal_r, x0, method='SLSQP', 
                           bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': constraints_equal_r},
                           options={'maxiter': 8000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement with LP
    if best_centers is not None:
        for _ in range(20):
            pert = best_centers + np.random.normal(0, 0.003, best_centers.shape)
            pert = np.clip(pert, 0.02, 0.98)
            r_lp = solve_lp_radii(pert)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = pert.copy()
                    best_radii = r_lp.copy()
                    
        # Phase 3: Joint center+radius optimization
        x0_j = np.zeros(3 * N_CIRCLES)
        x0_j[:N_CIRCLES] = best_centers[:, 0]
        x0_j[N_CIRCLES:2*N_CIRCLES] = best_centers[:, 1]
        x0_j[2*N_CIRCLES:] = best_radii
        
        bounds_j = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(1e-6, 0.25)] * N_CIRCLES
        try:
            res_j = minimize(objective_joint, x0_j, method='SLSQP',
                             bounds=bounds_j,
                             constraints={'type': 'ineq', 'fun': constraints_joint},
                             options={'maxiter': 5000, 'ftol': 1e-12})
            if np.isfinite(res_j.fun):
                c_j = res_j.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_j = res_j.x[2*N_CIRCLES:]
                s_j = np.sum(r_j)
                if s_j > best_sum:
                    best_sum = s_j
                    best_centers = c_j.copy()
                    best_radii = r_j.copy()
        except Exception:
            pass

    # Safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(N_CIRCLES):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999995
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
