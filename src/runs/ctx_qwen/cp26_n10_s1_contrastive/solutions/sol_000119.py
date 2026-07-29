# sol_000119 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000054 (state 94cc489d) state=e8d541ee sum of radii=2.628044 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[:N])

def constraints(v):
    """
    Computes inequality constraints g(v) >= 0.
    Uses (r, u, v) parameterization which automatically satisfies boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = v[:N]
    u = v[N:2*N]
    vt = v[2*N:3*N]
    
    # Map normalized u, v to actual coordinates within [r, 1-r]
    x = r + u * (1.0 - 2.0 * r)
    y = r + vt * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    
    # Squared sum of radii
    rs = r[:, None] + r[None, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def make_params_from_pts(pts):
    """Converts physical centers to (r, u, v) optimization parameters."""
    n = pts.shape[0]
    r = np.zeros(n)
    for i in range(n):
        dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        dists = np.linalg.norm(pts[i] - pts, axis=1)
        dists[i] = np.inf
        dp = np.min(dists)
        # Safe initial radius strictly inside feasible region
        r[i] = 0.85 * min(dw, dp/2.0)
    r = np.maximum(r, 1e-4)
    
    denom = 1.0 - 2.0 * r
    denom = np.clip(denom, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    
    return np.concatenate([r, u, v])

def gen_hex(row_counts, seed, rot, scale, jitter):
    """Generates a hexagonal lattice initialization with specified parameters."""
    np.random.seed(seed)
    pts = []
    r_e = 0.09
    y = r_e
    row = 0
    for cnt in row_counts:
        x_s = r_e if row % 2 == 0 else 2.0 * r_e
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x_s, y])
            x_s += 2.0 * r_e
        y += np.sqrt(3.0) * r_e
        row += 1
    pts = np.array(pts[:N])
    
    # Center and scale
    pts = (pts - 0.5) * scale + 0.5
    
    # Rotate
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        M = np.array([[c, -s], [s, c]])
        pts = pts @ M.T
        
    # Add jitter
    pts += np.random.uniform(-jitter, jitter, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def solve_radii_lp(centers):
    """Given fixed centers, solves LP to find optimal radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= dist_to_wall
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0-x, y, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    diff = centers[:, None, :] - centers[None, :, :]
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
    bounds = [(0.0, None)]*n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-5)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Diverse hexagonal patterns to capture different boundary alignments
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [7,6,6,7], [8,6,6,6], [6,4,6,5,5], [4,5,6,5,6],
        [7,5,5,5,4], [4,5,5,5,7], [6,6,6,4,4], [8,8,5,5]
    ]
    
    inits = []
    for pat in patterns:
        for s in range(6):
            rot = np.random.uniform(-0.3, 0.3)
            scale = np.random.uniform(0.85, 1.15)
            jit = np.random.uniform(0.01, 0.04)
            inits.append(make_params_from_pts(gen_hex(pat, s, rot, scale, jit)))
            
    # Grid initializations
    for s in range(10):
        np.random.seed(s)
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(make_params_from_pts(pts))
        
    # Random strictly feasible initializations
    for s in range(15):
        np.random.seed(s+100)
        pts = np.random.rand(N, 2)
        inits.append(make_params_from_pts(pts))
        
    # Phase 1: Broad search from diverse initializations
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            if res.success:
                if np.min(constraints(res.x)) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_v is not None:
        for k in range(50):
            x0 = best_v.copy()
            # Perturb positions (u, v) more aggressively than radii
            x0[N:3*N] += np.random.uniform(-0.02, 0.02, 2*N)
            x0[:N] += np.random.uniform(-0.002, 0.002, N)
            
            x0[:N] = np.clip(x0[:N], 1e-6, 0.5)
            x0[N:3*N] = np.clip(x0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
                if res.success:
                    if np.min(constraints(res.x)) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_v = res.x.copy()
            except Exception:
                pass
                
        # Phase 2b: LP refinement on radii for fixed centers
        r_opt = best_v[:N]
        u_opt = best_v[N:2*N]
        v_opt = best_v[2*N:3*N]
        x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
        y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
        centers_lp = np.column_stack((x_opt, y_opt))
        
        r_lp = solve_radii_lp(centers_lp)
        if np.sum(r_lp) > best_sum:
            # Re-parameterize with optimally computed radii
            denom = 1.0 - 2.0 * r_lp
            denom = np.clip(denom, 1e-6, 1.0)
            u_new = np.clip((centers_lp[:, 0] - r_lp) / denom, 0.0, 1.0)
            v_new = np.clip((centers_lp[:, 1] - r_lp) / denom, 0.0, 1.0)
            best_v = np.concatenate([r_lp, u_new, v_new])
            best_sum = np.sum(r_lp)
            
            # One more optimization step after LP update to relax positions
            try:
                res = minimize(objective, best_v, method='SLSQP', bounds=bounds_opt,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
                if res.success and np.min(constraints(res.x)) >= -1e-7:
                    if -res.fun > best_sum:
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_v, method='SLSQP', bounds=bounds_opt,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
            if np.min(constraints(res.x)) >= -1e-8:
                best_v = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback valid configuration
    if best_v is None:
        fallback_pts = np.zeros((N, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                fallback_pts[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        fallback_pts[25] = [0.5, 0.5]
        best_v = make_params_from_pts(fallback_pts)
        best_sum = np.sum(best_v[:N])
        
    # Reconstruct centers from optimized parameters
    r_opt = best_v[:N]
    u_opt = best_v[N:2*N]
    v_opt = best_v[2*N:3*N]
    
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
