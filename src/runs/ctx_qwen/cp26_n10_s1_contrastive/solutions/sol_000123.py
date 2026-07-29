# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000054 (state 94cc489d) state=a85a7f81 sum of radii=2.634292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(params):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraint_func(params):
    """
    Computes inequality constraints g(params) >= 0.
    Only pairwise non-overlap constraints are needed as boundaries are handled by parameterization.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def make_hex_init(seed, rot, scale):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.09
    y = r_est
    row = 0
    while len(pts) < N:
        x_off = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_off
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        pts -= pts.mean(axis=0)
        pts += 0.5
        
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_pair = np.min(dists, axis=1)
    d_wall = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]), np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    r = 0.9 * np.minimum(d_wall, d_pair / 2.0)
    
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def make_force_init(seed, steps=400):
    """Generates a strictly feasible initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    r = np.full(N, 0.05)
    
    for s in range(steps):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-8)
        rep_mag = 1.0 / (dists**2)
        np.fill_diagonal(rep_mag, 0.0)
        forces = np.sum(rep_mag[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
        
        for d in range(2):
            forces[:, d] += 10.0 * np.maximum(0, 0.05 - pts[:, d])
            forces[:, d] -= 10.0 * np.maximum(0, pts[:, d] - 0.95)
            
        pts += 0.005 * forces * (1.0 - s / float(steps))
        pts = np.clip(pts, 0.01, 0.99)
        
        np.fill_diagonal(dists, np.inf)
        d_pair = np.min(dists, axis=1)
        d_wall = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]), np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
        r = 0.95 * np.minimum(d_wall, d_pair / 2.0)
        
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def make_grid_init(seed):
    """Generates a perturbed grid initialization."""
    rng = np.random.RandomState(seed)
    pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    pts = np.vstack([pts, [0.5, 0.5]])
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_pair = np.min(dists, axis=1)
    d_wall = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]), np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    r = 0.9 * np.minimum(d_wall, d_pair / 2.0)
    
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = [(1e-5, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    
    inits = []
    
    # 1. Hexagonal lattice variations
    for s in range(25):
        rot = np.random.uniform(-0.35, 0.35)
        scale = np.random.uniform(0.80, 1.20)
        inits.append(make_hex_init(s, rot, scale))
        
    # 2. Force-directed layouts
    for s in range(20):
        inits.append(make_force_init(s))
        
    # 3. Perturbed grids
    for s in range(15):
        inits.append(make_grid_init(s))
        
    # Phase 1: Broad search from diverse initializations
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12})
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = np.sum(res.x[:N])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Intensive local refinement to escape local minima
    if best_vars is not None:
        for k in range(50):
            x0 = best_vars.copy()
            x0[:N] += np.random.uniform(-0.001, 0.001, N)
            x0[N:2*N] += np.random.uniform(-0.02, 0.02, N)
            x0[2*N:3*N] += np.random.uniform(-0.02, 0.02, N)
            x0[:N] = np.clip(x0[:N], 1e-5, 0.5)
            x0[N:3*N] = np.clip(x0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
                if res.success:
                    c_val = constraint_func(res.x)
                    if np.min(c_val) >= -1e-7:
                        s_val = np.sum(res.x[:N])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                best_vars = res.x
        except Exception:
            pass

    # Fallback configuration (should rarely be reached)
    if best_vars is None:
        best_vars = make_force_init(42)
        
    r_opt = best_vars[:N]
    u_opt = best_vars[N:2*N]
    v_opt = best_vars[2*N:3*N]
    
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, np.maximum(r_opt, 0.0), float(np.sum(r_opt))
