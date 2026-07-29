# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000105 (state 007a7b0d) state=9f1de13e sum of radii=2.620921 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[:N])

def constraints(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Only pairwise non-overlap constraints are enforced.
    Boundary constraints are handled analytically by the parameterization.
    """
    r = vars_vec[:N]
    u = vars_vec[N:2*N]
    v = vars_vec[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist_sq = dx**2 + dy**2
    r_sum = r[i_idx] + r[j_idx]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq - r_sum**2

def make_hex_init(seed, rot, scale):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    while len(pts) < N:
        shift = r_est if row % 2 == 1 else 0.0
        x = r_est + shift
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
        
    centers = np.array(pts[:N])
    centers -= 0.5
    centers *= scale
    centers += 0.5
    
    if abs(rot) > 1e-5:
        c, s = np.cos(rot), np.sin(rot)
        mat = np.array([[c, -s], [s, c]])
        centers = (centers - 0.5) @ mat.T + 0.5
        
    centers += rng.uniform(-0.015, 0.015, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r = np.full(N, 0.08)
    denom = np.clip(1.0 - 2.0 * r, 1e-6, None)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def make_force_init(seed):
    """Generates a strictly feasible initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    r = np.full(N, 0.05)
    
    for _ in range(200):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-6)
        rep = 1.0 / (dists**2)
        np.fill_diagonal(rep, 0.0)
        f = np.sum(rep[:, :, None] * diff / dists[:, :, None], axis=1)
        
        for d in range(2):
            f[:, d] += 10.0 * np.maximum(0, r - pts[:, d])
            f[:, d] -= 10.0 * np.maximum(0, pts[:, d] - (1.0 - r))
            
        pts += 0.001 * f
        pts = np.clip(pts, 0.02, 0.98)
        
    r = np.full(N, 0.08)
    denom = np.clip(1.0 - 2.0 * r, 1e-6, None)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def run_packing():
    bounds = [(1e-6, 0.45)] * N + [(0.0, 1.0)] * (2 * N)
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -np.inf
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse initializations
    inits = []
    for s in range(25):
        inits.append(make_hex_init(s, rot=rng.uniform(-0.25, 0.25), scale=rng.uniform(0.85, 1.15)))
    for s in range(15):
        inits.append(make_force_init(s))
    for s in range(20):
        r = rng.uniform(0.06, 0.11, N)
        u = rng.uniform(0.1, 0.9, N)
        v = rng.uniform(0.1, 0.9, N)
        inits.append(np.concatenate([r, u, v]))
        
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Multi-scale local perturbation refinement
    if best_vars is not None:
        for step_scale in [0.005, 0.003, 0.001]:
            for trial in range(40):
                x0 = best_vars.copy()
                x0 += rng.normal(0, step_scale, size=3*N)
                x0[:N] = np.clip(x0[:N], 1e-6, 0.45)
                x0[N:2*N] = np.clip(x0[N:2*N], 0.0, 1.0)
                x0[2*N:3*N] = np.clip(x0[2*N:3*N], 0.0, 1.0)
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
                    if res.success and np.min(constraints(res.x)) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
                except Exception:
                    pass
                    
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
            if res.success and np.min(constraints(res.x)) >= -1e-8:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass

    # Extract and return results
    r_opt = best_vars[:N]
    u_opt = best_vars[N:2*N]
    v_opt = best_vars[2*N:3*N]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack([x_opt, y_opt])
    radii = np.maximum(r_opt, 0.0)
    
    return centers, radii, float(best_sum)
