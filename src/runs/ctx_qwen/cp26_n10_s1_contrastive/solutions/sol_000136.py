# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000133 (state ddcde962) state=96feecc6 sum of radii=2.400000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_con = 4 * n + n * (n - 1) // 2
    A = np.zeros((num_con, n))
    b = np.zeros(num_con)
    k = 0
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        bounds_val = [x, 1.0 - x, y, 1.0 - y]
        for val in bounds_val:
            A[k, i] = 1.0
            b[k] = val
            k += 1
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            b[k] = dists[i, j]
            k += 1
            
    bounds = [(0.0, None)] * n
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, -res.fun
    return np.full(n, 1e-5), 1e-5

def obj_slqp(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[0::3])

def cons_slqp(v):
    """Inequality constraints: dist_sq >= (r_i + r_j)^2. Boundaries handled by parameterization."""
    r = v[0::3]
    u = v[1::3]
    w = v[2::3]
    
    x = r + u * (1.0 - 2.0 * r)
    y = r + w * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    return dist_sq[I_IDX, J_IDX] - r_sum_sq[I_IDX, J_IDX]

def make_vars(centers):
    """Map physical centers to (r, u, w) optimization parameters using LP radii."""
    r, _ = solve_lp(centers)
    r = np.clip(r, 1e-6, 0.499)
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    w = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    
    v = np.empty(3 * N)
    v[0::3] = r
    v[1::3] = u
    v[2::3] = w
    return v

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    
    # Hexagonal lattice patterns summing to 26
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [6,4,6,5,5], [5,6,4,6,5], [4,5,6,5,6], [6,6,5,5,4],
        [7,5,5,5,4], [4,5,5,5,7], [5,7,5,5,4], [6,5,5,5,5]
    ]
    
    for pat in patterns:
        for _ in range(4):
            pts = []
            r_est = 0.095
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = (r_idx % 2) * r_est
                x = r_est + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
                
            pts = np.array(pts[:N])
            pts = (pts - 0.5) * rng.uniform(0.85, 1.15) + 0.5
            
            rot = rng.uniform(-0.25, 0.25)
            c, s = np.cos(rot), np.sin(rot)
            pts = pts @ np.array([[c, -s], [s, c]])
            
            pts += rng.uniform(-0.01, 0.01, (N, 2))
            pts = np.clip(pts, 0.02, 0.98)
            inits.append(pts)
            
    # Random initializations
    for _ in range(10):
        pts = rng.uniform(0.05, 0.95, (N, 2))
        inits.append(pts)
        
    return inits

def run_packing():
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.49), (0.0, 1.0), (0.0, 1.0)] * N
    cons_slqp_dict = {'type': 'ineq', 'fun': cons_slqp}
    
    inits = generate_inits(rng)
    best_c = None
    best_r = None
    best_sum = -np.inf
    
    # Phase 1: Simulated Annealing Local Search on each initialization
    for pts in inits:
        r, curr_sum = solve_lp(pts)
        if curr_sum < 2.45: 
            continue # Skip clearly suboptimal starts
            
        centers = pts.copy()
        temp = 0.06
        step = 0.035
        best_c_loc = centers.copy()
        best_r_loc = r.copy()
        best_sum_loc = curr_sum
        curr_c = centers.copy()
        
        for it in range(600):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            
            direction = rng.standard_normal(2)
            direction /= np.linalg.norm(direction) + 1e-9
            curr_c[idx] = np.clip(curr_c[idx] + step * direction, 0.005, 0.995)
            
            nr, ns = solve_lp(curr_c)
            delta = ns - curr_sum
            
            if delta > 0 or (temp > 1e-6 and rng.random() < np.exp(delta / temp)):
                curr_sum = ns
                if ns > best_sum_loc:
                    best_sum_loc = ns
                    best_c_loc = curr_c.copy()
                    best_r_loc = nr.copy()
            else:
                curr_c[idx] = old
                
            temp *= 0.9994
            step = max(0.001, step * 0.9996)
            
        if best_sum_loc > best_sum:
            best_sum = best_sum_loc
            best_c = best_c_loc
            best_r = best_r_loc
            
    # Phase 2: SLSQP refinement on the best configuration found
    if best_c is not None:
        v0 = make_vars(best_c)
        v0[0::3] *= 0.99
        v0[0::3] = np.clip(v0[0::3], 1e-6, 0.49)
        
        try:
            res = minimize(obj_slqp, v0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp_dict, 
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                if np.min(cons_slqp(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        r_opt = res.x[0::3]
                        u_opt = res.x[1::3]
                        w_opt = res.x[2::3]
                        best_c = np.column_stack((r_opt + u_opt * (1.0 - 2.0 * r_opt), 
                                                  r_opt + w_opt * (1.0 - 2.0 * r_opt)))
                        best_r = r_opt
        except Exception:
            pass
            
        # Phase 3: Perturbation + SLSQP to escape local minima
        for _ in range(15):
            vp = v0.copy()
            vp[1::3] += rng.normal(0, 0.04, N)
            vp[2::3] += rng.normal(0, 0.04, N)
            vp[0::3] += rng.normal(0, 0.002, N)
            
            # Apply bounds safely
            vp = np.clip(vp, 0.0, 1.0)
            vp[0::3] = np.clip(vp[0::3], 1e-6, 0.49)
            
            try:
                res = minimize(obj_slqp, vp, method='SLSQP', bounds=bounds_slqp,
                               constraints=cons_slqp_dict,
                               options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                if res.success and np.min(cons_slqp(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        r_opt = res.x[0::3]
                        u_opt = res.x[1::3]
                        w_opt = res.x[2::3]
                        best_c = np.column_stack((r_opt + u_opt * (1.0 - 2.0 * r_opt), 
                                                  r_opt + w_opt * (1.0 - 2.0 * r_opt)))
                        best_r = r_opt
            except Exception:
                pass
                
        # Phase 4: High-precision final polish
        v0 = make_vars(best_c)
        v0[0::3] *= 0.995
        v0[0::3] = np.clip(v0[0::3], 1e-6, 0.49)
        try:
            res = minimize(obj_slqp, v0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp_dict,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-9:
                r_opt = res.x[0::3]
                u_opt = res.x[1::3]
                w_opt = res.x[2::3]
                best_c = np.column_stack((r_opt + u_opt * (1.0 - 2.0 * r_opt), 
                                          r_opt + w_opt * (1.0 - 2.0 * r_opt)))
                best_r = r_opt
                best_sum = np.sum(best_r)
        except Exception:
            pass

    # Fallback configuration (should rarely be reached)
    if best_c is None:
        fallback_pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        best_c = fallback_pts
        best_r, best_sum = solve_lp(best_c)
        
    return best_c, np.maximum(best_r, 0.0), float(best_sum)
