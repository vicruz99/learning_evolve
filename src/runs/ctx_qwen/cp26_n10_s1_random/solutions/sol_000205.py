# sol_000205 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000197 (state 20ef424a) state=a61f70e4 sum of radii=1.729152 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)

def obj_max_min(vars):
    """Objective: minimize negative R (maximize equal radius)."""
    return -vars[2*N]

def cons_max_min(vars):
    """Constraints for equal radius packing: boundary and pairwise clearance >= R."""
    x = vars[:N]
    y = vars[N:2*N]
    R = vars[2*N]
    c = np.empty(4*N + N*(N-1)//2)
    c[:N] = x - R
    c[N:2*N] = 1.0 - x - R
    c[2*N:3*N] = y - R
    c[3*N:4*N] = 1.0 - y - R
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    c[4*N:] = (dx[TRIU_I, TRIU_J]**2 + dy[TRIU_I, TRIU_J]**2) - 4.0*R**2
    return c

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = N
    c_obj = -np.ones(n)
    bounds = []
    b_ub_list = []
    A_rows = []
    
    for i in range(n):
        x, y = centers[i]
        w = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(w, 1e-9)))
        
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_ub_list.append(d)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_ub_list)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 1e-6

def get_hex_centers(row_counts, r_init=0.1):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r_init
    for idx, cnt in enumerate(row_counts):
        shift = r_init if idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
    while len(pts) < N:
        pts.append([0.5, 0.5])
    return np.array(pts[:N])

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint center/radius optimization."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    c = np.empty(4*N + N*(N-1)//2)
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    rs = r[:, None] + r[None, :]
    c[4*N:] = (dx[TRIU_I, TRIU_J]**2 + dy[TRIU_I, TRIU_J]**2) - rs[TRIU_I, TRIU_J]**2
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_R = 0.0
    best_centers_eq = None
    
    # Phase 1: Maximize equal radius using SLSQP from diverse starts
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [5, 6, 5, 5, 5], [6, 5, 5, 5, 5]
    ]
    
    starts = []
    for pat in row_patterns:
        c = get_hex_centers(pat, 0.10)
        starts.append(c)
        for _ in range(3):
            cp = c + rng.uniform(-0.02, 0.02, (N, 2))
            starts.append(np.clip(cp, 0.05, 0.95))
            
    for _ in range(5):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    cons_dict = {'type': 'ineq', 'fun': cons_max_min}
    bounds_mm = [(0.0, 1.0)]*(2*N) + [(0.05, 0.15)]
    
    for cfg in starts:
        d_wall = np.minimum(np.minimum(cfg[:,0], 1.0-cfg[:,0]), np.minimum(cfg[:,1], 1.0-cfg[:,1]))
        diff = cfg[:, None, :] - cfg[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        d_pair = np.min(dists) / 2.0
        R_init = min(np.min(d_wall), d_pair) * 0.95
        
        x0 = np.concatenate([cfg.flatten(), [R_init]])
        
        try:
            res = minimize(obj_max_min, x0, method='SLSQP', bounds=bounds_mm,
                           constraints=cons_dict, options={'maxiter': 4000, 'ftol': 1e-12})
            if np.isfinite(res.fun) and res.x[2*N] > best_R:
                best_R = res.x[2*N]
                best_centers_eq = res.x[:2*N].reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_centers_eq is None:
        best_centers_eq = get_hex_centers([5, 6, 5, 6, 4], 0.10)
        
    # Phase 2: LP refinement on equal radius centers
    radii_lp, sum_lp = solve_lp(best_centers_eq)
    best_centers = best_centers_eq.copy()
    best_radii = radii_lp.copy()
    best_sum = sum_lp
    
    # Phase 3: Stochastic Hill Climbing on centers
    for step in range(500):
        i = rng.integers(N)
        old_c = best_centers[i].copy()
        step_sz = 0.02 * (0.995 ** step)
        best_centers[i] += rng.uniform(-step_sz, step_sz, 2)
        best_centers[i] = np.clip(best_centers[i], 1e-4, 1.0 - 1e-4)
        
        r_try, s_try = solve_lp(best_centers)
        if s_try > best_sum:
            best_sum = s_try
            best_radii = r_try.copy()
        else:
            best_centers[i] = old_c
            
    # Phase 4: Joint SLSQP refinement
    x0_joint = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii])
    bounds_joint = [(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N
    cons_j = {'type': 'ineq', 'fun': cons_joint}
    
    for _ in range(5):
        x0_pert = x0_joint.copy()
        x0_pert[:2*N] += rng.uniform(-0.005, 0.005, 2*N)
        x0_pert[:2*N] = np.clip(x0_pert[:2*N], 0.01, 0.99)
        x0_pert[2*N:] = np.clip(x0_pert[2*N:], 1e-6, 0.5)
        
        try:
            res_j = minimize(obj_joint, x0_pert, method='SLSQP', bounds=bounds_joint,
                             constraints=cons_j, options={'maxiter': 3000, 'ftol': 1e-12})
            if np.isfinite(res_j.fun):
                c_val = cons_joint(res_j.x)
                if np.min(c_val) > -1e-7:
                    c_opt = res_j.x[:2*N].reshape(N, 2)
                    r_lp2, s_lp2 = solve_lp(c_opt)
                    if s_lp2 > best_sum:
                        best_sum = s_lp2
                        best_centers = c_opt.copy()
                        best_radii = r_lp2.copy()
        except Exception:
            pass

    # Final safety scaling
    scale = 1.0
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-12: continue
        scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        
    for k in range(len(TRIU_I)):
        i, j = TRIU_I[k], TRIU_J[k]
        d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
        rs = best_radii[i] + best_radii[j]
        if rs < 1e-12: continue
        scale = min(scale, d / rs)
        
    best_radii *= scale * 0.9999995
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(final_sum)
