# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000059 (state 3e3cfdc0) state=67e5b4e6 sum of radii=2.626247 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraint_func(params):
    """
    Inequality constraints: g(params) >= 0.
    Parameterization automatically satisfies boundary constraints.
    Only pairwise non-overlap is enforced.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Map parameters back to physical coordinates
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dist_sq = dx * dx + dy * dy
    r_sum = r[I_IDX] + r[J_IDX]
    
    return dist_sq - r_sum * r_sum

def get_params_from_physical(centers, radii):
    """Convert physical (centers, radii) to optimization parameters (r, u, v)."""
    r = radii.copy()
    denom = 1.0 - 2.0 * r
    denom = np.clip(denom, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def get_physical_from_params(params):
    """Convert optimization parameters (r, u, v) back to physical (centers, radii)."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack((x, y)), r

def generate_init(seed, rot, scale, jitter):
    """Generate a strictly feasible initial configuration from a rotated/scaled hex lattice."""
    np.random.seed(seed)
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    while len(pts) < N:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_start
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    
    # Center, scale, and rotate
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        pts -= pts.mean(axis=0)
        pts += 0.5
        
    pts += np.random.uniform(-jitter, jitter, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    
    # Compute strictly feasible initial radii
    r = np.zeros(N)
    for i in range(N):
        dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        dm = np.inf
        for j in range(N):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
                if d < dm:
                    dm = d
        r[i] = 0.85 * min(dw, dm / 2.0)
        
    return get_params_from_physical(pts, r)

def run_packing():
    """
    Solves the circle packing problem for N=26 in a unit square.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * (2 * N)
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_params = None
    best_sum = -np.inf
    
    # Phase 1: Diverse global search from structured initializations
    inits = []
    for s in range(35):
        rot = np.random.uniform(-0.35, 0.35)
        scale = np.random.uniform(0.85, 1.15)
        jitt = np.random.uniform(0.005, 0.04)
        inits.append(generate_init(s, rot, scale, jitt))
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_params is not None:
        for k in range(45):
            x0 = best_params.copy()
            # Perturb positions more aggressively, radii conservatively
            x0[:N] += np.random.uniform(-0.002, 0.002, N)
            x0[N:3*N] += np.random.uniform(-0.015, 0.015, 2 * N)
            x0[:N] = np.clip(x0[:N], 1e-6, 0.5)
            x0[N:] = np.clip(x0[N:], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = constraint_func(res.x)
                    if np.min(c_val) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_params = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_params, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                best_params = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback configuration (should rarely be reached)
    if best_params is None:
        best_params = generate_init(0, 0.0, 1.0, 0.01)
        best_sum = np.sum(best_params[:N])
        
    centers, radii = get_physical_from_params(best_params)
    return centers, radii, float(np.sum(radii))
