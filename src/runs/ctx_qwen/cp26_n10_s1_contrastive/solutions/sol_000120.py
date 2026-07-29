# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000054 (state 94cc489d) state=59dc24f4 sum of radii=2.624849 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
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
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.full(n, 1e-5), 2.6e-4

def get_params_from_centers(centers, scale_r=0.99):
    """Converts physical centers to (r, u, v) optimization parameters using LP radii."""
    r, _ = solve_lp_radii(centers)
    r = r * scale_r
    r = np.clip(r, 1e-6, 0.49)
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def objective(params):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraints(params):
    """
    Computes inequality constraints g(params) >= 0.
    Uses parameterization to automatically satisfy boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:]
    
    # Map normalized u, v to actual coordinates within [r, 1-r]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Vectorized pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = [(1e-6, 0.49)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_params = None
    best_sum = -np.inf
    
    # Phase 1: Diverse global search with LP-warmed starts
    inits = []
    for s in range(30):
        rot = np.random.uniform(-0.3, 0.3)
        scale = np.random.uniform(0.85, 1.15)
        
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
            c, s_val = np.cos(rot), np.sin(rot)
            pts = (pts - 0.5) @ np.array([[c, -s_val], [s_val, c]]) + 0.5
            
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(pts)
        
    # Add perturbed grid initialization
    grid = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
    grid += np.random.uniform(-0.02, 0.02, grid.shape)
    grid = np.clip(grid, 0.02, 0.98)
    inits.append(grid)
    
    for pts in inits:
        p0 = get_params_from_centers(pts, scale_r=0.99)
        try:
            res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13})
            if res.success:
                if np.min(constraints(res.x)) >= -1e-7:
                    s_val = np.sum(res.x[:N])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_params = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_params is not None:
        for k in range(40):
            p0 = best_params.copy()
            p0[N:] += np.random.uniform(-0.02, 0.02, 2 * N)
            p0[:N] += np.random.uniform(-0.002, 0.002, N)
            p0 = np.clip(p0, 0.0, 1.0)
            p0[:N] = np.clip(p0[:N], 1e-6, 0.49)
            
            try:
                res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-13})
                if res.success:
                    if np.min(constraints(res.x)) >= -1e-7:
                        s_val = np.sum(res.x[:N])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_params = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_params, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-14})
            if res.success and np.min(constraints(res.x)) >= -1e-8:
                best_params = res.x
                best_sum = np.sum(best_params[:N])
        except Exception:
            pass

    # Fallback valid configuration (should rarely be reached)
    if best_params is None:
        pts = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                               np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        r_f = np.full(N, 0.04)
        best_params = np.concatenate([r_f, np.full(N, 0.5), np.full(N, 0.5)])
        best_sum = np.sum(r_f)
        
    # Reconstruct centers from optimized parameters
    r_opt = best_params[:N]
    u_opt = best_params[N:2*N]
    v_opt = best_params[2*N:]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
