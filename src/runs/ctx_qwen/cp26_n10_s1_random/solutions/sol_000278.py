# sol_000278 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000269 (state 0e622564) state=654a75fe sum of radii=2.512457 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def solve_lp_fast(centers, A, b_ub, idx_i, idx_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(w, 1e-12)) for w in wall]
    
    # Update pairwise distance constraints in-place
    diff = centers[idx_i] - centers[idx_j]
    b_ub[:] = np.sqrt(np.sum(diff**2, axis=1))
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'maxiter': 1000})
        if res.success and np.isfinite(res.fun):
            r = np.maximum(res.x, 0.0)
            return r, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def clearance_obj(x, n):
    """Objective for Nelder-Mead: maximize minimum clearance (equal-radius proxy)."""
    c = x.reshape(n, 2)
    d_w = np.min(np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                            np.minimum(c[:, 1], 1.0 - c[:, 1])))
    diff = c[:, None, :] - c[None, :, :]
    # Add large diagonal to ignore self-distance
    d_p = np.min(np.sqrt(np.sum(diff**2, axis=2) + np.eye(n) * 1e12)) / 2.0
    return -min(d_w, d_p)

def joint_obj(x, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def joint_cons(x, n, idx_i, idx_j):
    """Inequality constraints >= 0 for SLSQP joint optimization."""
    cx, cy, r = x[:n], x[n:2 * n], x[2 * n:]
    con = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    dr = r[idx_i] + r[idx_j]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    # Precompute pairwise indices and constraint matrix structure
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A = np.zeros((m, n))
    A[np.arange(m), idx_i] = 1.0
    A[np.arange(m), idx_j] = 1.0
    b_ub = np.zeros(m)
    
    # Generate diverse initial hexagonal configurations
    configs = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [7, 5, 5, 5, 4]
    ]
    
    for pat in patterns:
        if sum(pat) < n: continue
        pts = []
        y = 0.10
        for r_idx, cnt in enumerate(pat):
            shift = 0.10 if r_idx % 2 == 1 else 0.0
            x = 0.10 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 0.20
            y += 0.10 * np.sqrt(3)
        configs.append(np.array(pts[:n]))
        
    # Add perturbed variants to break symmetry
    for cfg in configs[:2]:
        for _ in range(10):
            p = cfg + rng.uniform(-0.025, 0.025, (n, 2))
            configs.append(np.clip(p, 0.05, 0.95))
            
    best_c, best_r, best_s = None, None, 0.0
    
    # Phase 1: Clearance Maximization + LP Expansion
    for cfg in configs:
        try:
            res = minimize(clearance_obj, cfg.flatten(), args=(n,), method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            c_opt = np.clip(res.x.reshape(n, 2), 0.01, 0.99)
            r, s = solve_lp_fast(c_opt, A, b_ub, idx_i, idx_j)
            if s > best_s:
                best_c, best_r, best_s = c_opt.copy(), r.copy(), s
        except Exception:
            pass
            
    # Phase 2: Joint SLSQP Polish
    if best_c is not None:
        bounds_v = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
        x0 = np.concatenate([best_c[:, 0], best_c[:, 1], best_r * 0.97])
        try:
            res = minimize(joint_obj, x0, args=(n,), method='SLSQP',
                          bounds=bounds_v,
                          constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n, idx_i, idx_j)},
                          options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                c_s = np.column_stack((res.x[:n], res.x[n:2 * n]))
                r_s, s_s = solve_lp_fast(c_s, A, b_ub, idx_i, idx_j)
                if s_s > best_s:
                    best_c, best_r, best_s = c_s.copy(), r_s.copy(), s_s
        except Exception:
            pass

    # Phase 3: Fine-Grained Simulated Annealing on Centers
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        
        for it in range(8000):
            temp = 0.006 * (1.0 - it / 8000.0)**0.5 + 1e-5
            # Perturb 1 or 2 circles simultaneously
            k = rng.choice([1, 2], p=[0.7, 0.3])
            idx = rng.choice(n, size=k, replace=False)
            old = curr_c[idx].copy()
            
            step = temp * 2.0
            curr_c[idx] += rng.uniform(-step, step, (k, 2))
            curr_c[idx] = np.clip(curr_c[idx], 1e-4, 1.0 - 1e-4)
            
            r_new, s_new = solve_lp_fast(curr_c, A, b_ub, idx_i, idx_j)
            delta = s_new - curr_s
            
            # Accept if improves, or with probability based on temperature
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-9)):
                curr_r = r_new
                curr_s = s_new
            else:
                curr_c[idx] = old
                
            # Periodic local polish to settle into basins
            if it % 800 == 799:
                try:
                    res_loc = minimize(clearance_obj, curr_c.flatten(), args=(n,), method='Nelder-Mead',
                                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                    c_pol = np.clip(res_loc.x.reshape(n, 2), 1e-4, 1.0 - 1e-4)
                    r_pol, s_pol = solve_lp_fast(c_pol, A, b_ub, idx_i, idx_j)
                    if s_pol > curr_s:
                        curr_c = c_pol.copy()
                        curr_r = r_pol.copy()
                        curr_s = s_pol
                except Exception:
                    pass
                    
        best_c, best_r, best_s = curr_c, curr_r, curr_s

    # Phase 4: Strict Numerical Safety Scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    diff = best_c[idx_i] - best_c[idx_j]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_pair = best_r[idx_i] + best_r[idx_j]
    min_ratio = np.min(dists / np.maximum(r_pair, 1e-12))
    scale = min(scale, min_ratio)
    
    # Apply tight safety margin to strictly satisfy 1e-12 validator tolerance
    best_r *= max(0.999999999999, scale - 1e-13)
    best_s = float(np.sum(best_r))
    
    # Fallback safety net (should not be triggered)
    if best_c is None:
        best_c = configs[0]
        best_r = np.full(n, 0.085)
        best_s = float(np.sum(best_r))
        
    return best_c, best_r, best_s
