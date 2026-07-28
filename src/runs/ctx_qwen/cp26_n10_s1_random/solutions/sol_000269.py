# sol_000269 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000250 (state 61d3a642) state=0e622564 sum of radii=2.635980 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

def get_pairwise_indices(n):
    """Returns indices for upper triangle of distance matrix."""
    return np.triu_indices(n, k=1)

def build_lp_matrix(n, idx_i, idx_j):
    """Constructs the constraint matrix A for the LP: r_i + r_j <= dist_ij."""
    m = len(idx_i)
    A = np.zeros((m, n))
    A[np.arange(m), idx_i] = 1.0
    A[np.arange(m), idx_j] = 1.0
    return A

def solve_lp(centers, A, idx_i, idx_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    wall = np.minimum(np.minimum(centers[:,0], 1.0 - centers[:,0]), 
                      np.minimum(centers[:,1], 1.0 - centers[:,1]))
    bounds = [(0.0, max(w, 1e-12)) for w in wall]
    
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    diff = centers[idx_i] - centers[idx_j]
    b_ub = np.sqrt(np.sum(diff**2, axis=1))
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            r = np.maximum(res.x, 0.0)
            return r, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def clearance_obj(x_flat, n):
    """Objective for Nelder-Mead: maximize minimum clearance (equal-radius packing proxy)."""
    c = x_flat.reshape(n, 2)
    # Distance to walls
    d_wall = np.minimum(np.minimum(c[:,0], 1.0 - c[:,0]), 
                        np.minimum(c[:,1], 1.0 - c[:,1]))
    # Distance to other circles
    diff = c[:, None, :] - c[None, :, :]
    d_pair = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d_pair, np.inf)
    
    # Minimum clearance is limited by walls and half the pairwise distances
    min_d = np.min(np.concatenate([d_wall, d_pair.min(axis=1) / 2.0]))
    return -min_d  # Minimize negative clearance => Maximize clearance

def joint_obj(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def joint_cons(v, n, idx_i, idx_j):
    """Inequality constraints >= 0 for SLSQP joint optimization."""
    cx, cy, r = v[:n], v[n:2*n], v[2*n:]
    # Boundary constraints
    con = [cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r]
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    dr = r[idx_i] + r[idx_j]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    idx_i, idx_j = get_pairwise_indices(n)
    A = build_lp_matrix(n, idx_i, idx_j)
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], 
        [4,6,6,6,4], [7,5,5,5,4], [5,7,4,5,5], [6,4,6,5,5]
    ]
    
    for pat in patterns:
        if sum(pat) < n: continue
        pts = []
        y = 0.10
        for i, cnt in enumerate(pat):
            shift = 0.10 if i % 2 == 1 else 0.0
            x = 0.10 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 0.20
            y += 0.10 * np.sqrt(3)
        c = np.array(pts[:n])
        configs.append(c)
        # Add perturbed variants to break symmetry
        for _ in range(2):
            cp = c + rng.uniform(-0.025, 0.025, c.shape)
            configs.append(np.clip(cp, 0.05, 0.95))
            
    # Add purely random starts
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Clearance Maximization + LP
    for cfg in configs:
        try:
            res_nm = minimize(clearance_obj, cfg.flatten(), args=(n,), method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-10})
            c_opt = np.clip(res_nm.x.reshape(n, 2), 0.01, 0.99)
            r_lp, s_lp = solve_lp(c_opt, A, idx_i, idx_j)
            if s_lp > best_sum:
                best_sum = s_lp
                best_c = c_opt.copy()
                best_r = r_lp.copy()
        except Exception:
            pass
            
    # Phase 2: SLSQP Joint Polish
    if best_c is not None:
        bounds_v = [(0.0, 1.0)]*(2*n) + [(1e-5, 0.5)]*n
        x0 = np.concatenate([best_c[:,0], best_c[:,1], best_r * 0.98])
        
        try:
            res_slqp = minimize(joint_obj, x0, args=(n,), method='SLSQP',
                                bounds=bounds_v,
                                constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n, idx_i, idx_j)},
                                options={'maxiter': 3000, 'ftol': 1e-13})
            if np.isfinite(res_slqp.fun):
                c_s = np.column_stack((res_slqp.x[:n], res_slqp.x[n:2*n]))
                r_s, s_s = solve_lp(c_s, A, idx_i, idx_j)
                if r_s is not None and s_s > best_sum:
                    best_sum = s_s
                    best_c = c_s.copy()
                    best_r = r_s.copy()
        except Exception:
            pass

    # Phase 3: Multi-Circle Hill Climbing with Local Refinement
    curr_c = best_c.copy()
    curr_r = best_r.copy()
    curr_s = best_sum
    
    for it in range(1500):
        scale = 0.012 * (1.0 - it / 1500.0)**0.4
        # Perturb 1 to 3 circles simultaneously
        num_pert = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        idx = rng.choice(n, size=num_pert, replace=False)
        old = curr_c[idx].copy()
        
        curr_c[idx] += rng.uniform(-scale, scale, (num_pert, 2))
        curr_c[idx] = np.clip(curr_c[idx], 1e-4, 1.0 - 1e-4)
        
        r_new, s_new = solve_lp(curr_c, A, idx_i, idx_j)
        if r_new is not None and s_new > curr_s + 1e-7:
            curr_s = s_new
            curr_r = r_new.copy()
            
            # Local Nelder-Mead polish to settle the perturbation
            try:
                res_loc = minimize(clearance_obj, curr_c.flatten(), args=(n,), method='Nelder-Mead',
                                   options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-10})
                c_pol = np.clip(res_loc.x.reshape(n, 2), 0.01, 0.99)
                r_pol, s_pol = solve_lp(c_pol, A, idx_i, idx_j)
                if s_pol > curr_s:
                    curr_s = s_pol
                    curr_r = r_pol.copy()
                    curr_c = c_pol.copy()
            except Exception:
                pass
        else:
            curr_c[idx] = old
            
        best_c = curr_c
        best_r = curr_r
        best_sum = curr_s
        
    # Phase 4: Strict Numerical Safety Scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i,0], best_c[i,1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0 - x)/r, y/r, (1.0 - y)/r)
            
    diff = best_c[:, None, :] - best_c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = best_r[:, None] + best_r[None, :]
    min_ratio = np.min(dists[idx_i, idx_j] / np.maximum(r_pair[idx_i, idx_j], 1e-12))
    scale = min(scale, min_ratio)
    
    best_r *= scale * 0.999999
    best_sum = float(np.sum(best_r))
    
    # Fallback safety net
    if best_c is None:
        best_c = configs[0]
        best_r = np.full(n, 0.085)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, best_sum
