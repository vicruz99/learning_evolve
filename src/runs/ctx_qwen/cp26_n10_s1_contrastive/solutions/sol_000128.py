# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000059 (state 3e3cfdc0) state=505c2eb0 sum of radii=2.614985 correctness=1.0
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
    Computes inequality constraints g(params) >= 0.
    Uses (r, u, v) parameterization which automatically satisfies boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Map (r, u, v) to physical coordinates
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq[I_IDX, J_IDX] - r_sum_sq[I_IDX, J_IDX]

def make_hex_init(seed, rotation, scale):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    r_est = 0.095
    pts = []
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
    
    # Center, scale, rotate
    pts -= 0.5
    pts *= scale
    pts += 0.5
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    pts += rng.uniform(-0.01, 0.01, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def make_force_init(seed):
    """Generates a strictly feasible initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    r_curr = np.full(N, 0.05)
    
    for step in range(200):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-8)
        repulsion = 1.0 / dists**2
        np.fill_diagonal(repulsion, 0.0)
        forces = np.sum(repulsion[:, :, None] * diff / dists[:, :, None], axis=1)
        
        # Wall repulsion
        for d in range(2):
            forces[:, d] += 5.0 * np.maximum(0, r_curr - pts[:, d])
            forces[:, d] -= 5.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
            
        step_size = 0.005 * (0.99**step)
        pts += step_size * forces
        pts = np.clip(pts, 1e-5, 1.0 - 1e-5)
        
        # Update effective radii during simulation
        for i in range(N):
            dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            dm = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            r_curr[i] = 0.8 * min(dw, max(dm, 1e-6)/2.0)
            
    return pts

def to_params(centers, radii):
    """Maps physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def run_packing():
    np.random.seed(42)
    bounds = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    rng = np.random.RandomState(42)
    
    inits = []
    
    # 1. Hexagonal lattice variations (exploits optimal 2D packing geometry)
    for s in range(20):
        rot = rng.uniform(-0.3, 0.3)
        sc = rng.uniform(0.85, 1.15)
        pts = make_hex_init(s, rot, sc)
        r_vals = np.zeros(N)
        for i in range(N):
            dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            dm = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            r_vals[i] = 0.9 * min(dw, max(dm, 1e-6)/2.0)
        inits.append(to_params(pts, r_vals))
        
    # 2. Force-directed variations (finds natural high-density clusters)
    for s in range(15):
        pts = make_force_init(s)
        r_vals = np.zeros(N)
        for i in range(N):
            dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            dm = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            r_vals[i] = 0.9 * min(dw, max(dm, 1e-6)/2.0)
        inits.append(to_params(pts, r_vals))
        
    # Phase 1: Broad search from diverse initializations
    for p0 in inits:
        try:
            res = minimize(objective, p0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                if np.min(constraint_func(res.x)) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(35):
            p0 = best_vars.copy()
            p0[:N] += rng.uniform(-0.001, 0.001, N)
            p0[N:3*N] += rng.uniform(-0.01, 0.01, 2*N)
            p0[:N] = np.clip(p0[:N], 1e-6, 0.49)
            p0[N:3*N] = np.clip(p0[N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, p0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    if np.min(constraint_func(res.x)) >= -1e-7:
                        s_val = -res.fun
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass

    # Fallback configuration (should rarely be reached)
    if best_vars is None:
        fallback_pts = np.random.rand(N, 2)
        fallback_r = np.full(N, 0.04)
        best_vars = to_params(fallback_pts, fallback_r)
        best_sum = np.sum(fallback_r)

    # Reconstruct centers from optimized parameters
    r_opt = best_vars[:N]
    u_opt = best_vars[N:2*N]
    v_opt = best_vars[2*N:3*N]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    radii = np.maximum(r_opt, 0.0)
    
    return centers, radii, float(np.sum(radii))
