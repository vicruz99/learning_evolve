# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000094 (state 7fa10e6b) state=22dc88e6 sum of radii=2.630588 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp(centers):
    """Solves LP to find optimal radii for fixed centers. Returns (radii, sum_radii)."""
    n = centers.shape[0]
    num_constraints = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    k = 0
    
    # Precompute pairwise distances efficiently
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    c_obj = -np.ones(n)
    
    for i in range(n):
        x, y = centers[i]
        bounds_list = [x, 1.0 - x, y, 1.0 - y]
        for b in bounds_list:
            A_ub[k, i] = 1.0
            b_ub[k] = b
            k += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    bounds = [(0.0, None)] * n
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, -res.fun
    # Fallback to tiny radii if LP fails (should rarely happen for valid centers)
    return np.full(n, 1e-5), 2.6e-4

def objective_slqp(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def constraints_slqp(v):
    """Inequality constraints: boundary containment and pairwise non-overlap."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    c = []
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    c.append(dist2[i_idx, j_idx] - rs[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def generate_hex_centers(seed, rot=0.0, scale=1.0):
    """Generates a hexagonal lattice initialization with specified parameters."""
    np.random.seed(seed)
    rows_counts = [6, 5, 6, 5, 4]
    pts = []
    r_est = 0.1
    y = r_est
    for r_idx, cnt in enumerate(rows_counts):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            pts.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
        
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    
    if rot != 0.0:
        c_val, s_val = np.cos(rot), np.sin(rot)
        rot_mat = np.array([[c_val, -s_val], [s_val, c_val]])
        pts = pts @ rot_mat.T
        pts = pts - pts.mean(axis=0) + 0.5
        
    pts += np.random.randn(N, 2) * 0.02
    return np.clip(pts, 0.02, 0.98)

def run_packing():
    np.random.seed(42)
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    bounds_slqp = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_slqp = {'type': 'ineq', 'fun': constraints_slqp}
    
    # Phase 1: Diverse SLSQP starts
    starts = []
    for s in range(30):
        rot = np.random.uniform(-0.15, 0.15)
        sc = np.random.uniform(0.95, 1.05)
        starts.append(generate_hex_centers(s, rot, sc))
        
    for pts in starts:
        r_init, _ = solve_lp(pts)
        r_init = np.maximum(r_init * 0.995, 1e-6)
        v0 = np.zeros(3 * N)
        v0[0::3] = pts[:, 0]
        v0[1::3] = pts[:, 1]
        v0[2::3] = r_init
        
        try:
            res = minimize(objective_slqp, v0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 2000, 'ftol': 1e-13})
            if res.success:
                if np.min(constraints_slqp(res.x)) >= -1e-8:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_c = np.column_stack((res.x[0::3], res.x[1::3]))
                        best_r = res.x[2::3]
        except Exception:
            pass
            
    # Phase 2: Intensive LP-based Local Search with SA & Swaps
    if best_c is not None:
        centers = best_c.copy()
        radii, curr_sum = solve_lp(centers)
        best_lp_sum = curr_sum
        best_lp_c = centers.copy()
        best_lp_r = radii.copy()
        
        rng = np.random.RandomState(123)
        temp = 0.025
        step = 0.022
        
        for it in range(8000):
            op = rng.choice(['move', 'swap'])
            if op == 'move':
                idx = rng.randint(N)
                old = centers[idx].copy()
                centers[idx] += rng.normal(0, step, 2)
                centers[idx] = np.clip(centers[idx], 0.005, 0.995)
            else:
                i, j = rng.choice(N, 2, replace=False)
                centers[i], centers[j] = centers[j].copy(), centers[i].copy()
                
            r_new, sum_new = solve_lp(centers)
            delta = sum_new - curr_sum
            
            if delta > 0 or (temp > 1e-6 and rng.rand() < np.exp(delta / temp)):
                curr_sum = sum_new
                if sum_new > best_lp_sum:
                    best_lp_sum = sum_new
                    best_lp_c = centers.copy()
                    best_lp_r = r_new.copy()
            else:
                if op == 'move':
                    centers[idx] = old
                    
            temp *= 0.9992
            step = max(0.0015, step * 0.9994)
            
        best_c, best_r, best_sum = best_lp_c, best_lp_r, best_lp_sum
        
    # Phase 3: Rotation Escapes & Re-optimization
    if best_c is not None:
        for _ in range(15):
            rot = np.random.uniform(-0.1, 0.1)
            c_val, s_val = np.cos(rot), np.sin(rot)
            mat = np.array([[c_val, -s_val], [s_val, c_val]])
            c_rot = (best_c - 0.5) @ mat.T + 0.5
            c_rot = np.clip(c_rot, 0.01, 0.99)
            
            r_rot, _ = solve_lp(c_rot)
            v0 = np.zeros(3 * N)
            v0[0::3] = c_rot[:, 0]
            v0[1::3] = c_rot[:, 1]
            v0[2::3] = np.maximum(r_rot * 0.995, 1e-6)
            
            try:
                res_r = minimize(objective_slqp, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints=cons_slqp, options={'maxiter': 1500, 'ftol': 1e-13})
                if res_r.success:
                    if np.min(constraints_slqp(res_r.x)) >= -1e-8:
                        s_r = -res_r.fun
                        if s_r > best_sum:
                            best_sum = s_r
                            best_c = np.column_stack((res_r.x[0::3], res_r.x[1::3]))
                            best_r = res_r.x[2::3]
            except Exception:
                pass
                
    # Phase 4: High-precision Final Polish
    if best_c is not None:
        v0 = np.zeros(3 * N)
        v0[0::3] = best_c[:, 0]
        v0[1::3] = best_c[:, 1]
        v0[2::3] = best_r
        try:
            res_f = minimize(objective_slqp, v0, method='SLSQP', bounds=bounds_slqp,
                             constraints=cons_slqp, options={'maxiter': 4000, 'ftol': 1e-14})
            if res_f.success and np.min(constraints_slqp(res_f.x)) >= -1e-9:
                s_f = -res_f.fun
                if s_f > best_sum:
                    best_sum = s_f
                    best_c = np.column_stack((res_f.x[0::3], res_f.x[1::3]))
                    best_r = res_f.x[2::3]
        except Exception:
            pass
            
    # Fallback (should not be reached)
    if best_c is None:
        best_c = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                  np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        best_r = np.full(N, 0.04)
        best_sum = np.sum(best_r)
        
    return best_c, np.maximum(best_r, 0.0), float(best_sum)
