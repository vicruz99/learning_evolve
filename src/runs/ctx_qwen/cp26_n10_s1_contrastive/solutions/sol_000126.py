# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000059 (state 3e3cfdc0) state=fbc70012 sum of radii=2.628170 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Inequality constraints: dist_sq >= (r_i + r_j)^2.
    Boundary constraints are satisfied by the parameterization.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq[I_IDX, J_IDX] - r_sum_sq[I_IDX, J_IDX]

def make_vars_from_pts(pts):
    """Map physical centers to (r, u, v) optimization parameters with strict feasibility."""
    n = pts.shape[0]
    r = np.zeros(n)
    for i in range(n):
        dw = min(pts[i,0], 1.0 - pts[i,0], pts[i,1], 1.0 - pts[i,1])
        dm = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < dm:
                    dm = d
        # Safety factor ensures strict interior feasibility for SLSQP
        r[i] = max(1e-4, 0.85 * min(dw, dm / 2.0))
        
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((pts[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r) / denom, 0.0, 1.0)
    
    vars0 = np.empty(3 * n)
    vars0[0::3] = r
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def hex_init(n, r_est, rotation=0.0, scale=1.0, seed=0):
    """Generates a hexagonal lattice initialization with rotation, scaling, and jitter."""
    rng = np.random.RandomState(seed)
    pts = []
    y = r_est
    row = 0
    while len(pts) < n:
        shift = (row % 2) * r_est
        x = r_est + shift
        while x <= 1.0 - r_est and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:n])
    pts -= 0.5
    pts *= scale
    pts += 0.5
    
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        pts = pts @ np.array([[c, -s], [s, c]])
        
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(n, seed):
    """Generates a tightly packed initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    r_curr = np.full(n, 0.04)
    
    for step in range(600):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-8)
        rep_mag = 1.0 / dists**2
        np.fill_diagonal(rep_mag, 0.0)
        forces = np.sum(rep_mag[:, :, None] * diff / dists[:, :, None], axis=1)
        
        # Wall repulsion
        for d in range(2):
            forces[:, d] += 10.0 * np.maximum(0, r_curr - pts[:, d])
            forces[:, d] -= 10.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
            
        step_size = 0.008 * (0.998**step)
        pts += step_size * forces
        pts = np.clip(pts, 1e-4, 1.0 - 1e-4)
        
        # Adaptively update radii
        for i in range(n):
            d_wall = min(pts[i,0], 1.0 - pts[i,0], pts[i,1], 1.0 - pts[i,1])
            d_pair = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            d_pair = max(d_pair, 1e-6)
            r_curr[i] = 0.9 * min(d_wall, d_pair / 2.0)
            
    return pts

def run_packing():
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -1.0
    rng = np.random.RandomState(42)
    
    inits = []
    
    # 1. Force-directed initializations (excellent at finding tight, disordered packings)
    for s in range(15):
        pts = force_init(N, seed=s)
        inits.append(make_vars_from_pts(pts))
        
    # 2. Hexagonal lattice variations with random rotations and scales
    for s in range(25):
        rot = rng.uniform(-0.3, 0.3)
        sc = rng.uniform(0.85, 1.15)
        pts = hex_init(N, r_est=0.095, rotation=rot, scale=sc, seed=s)
        inits.append(make_vars_from_pts(pts))
        
    # 3. Perturbed grid layouts
    for s in range(5):
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        pts += rng.uniform(-0.03, 0.03, (N, 2))
        pts = np.clip(pts, 0.05, 0.95)
        inits.append(make_vars_from_pts(pts))
        
    # Phase 1: Broad search from diverse initializations
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                if np.min(constraint_func(res.x)) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(40):
            x0 = best_vars.copy()
            pert = rng.randn(3 * N)
            # Perturb positions more aggressively than radii
            x0[0::3] += pert[0::3] * 0.005
            x0[1::3] += pert[1::3] * 0.04
            x0[2::3] += pert[2::3] * 0.04
            
            x0[0::3] = np.clip(x0[0::3], 1e-6, 0.5)
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    if np.min(constraint_func(res.x)) >= -1e-7:
                        s = -res.fun
                        if s > best_sum:
                            best_sum = s
                            best_vars = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res.x)) >= -1e-8:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback configuration (should rarely be reached)
    if best_vars is None:
        pts = hex_init(N, 0.095)
        best_vars = make_vars_from_pts(pts)
        best_sum = np.sum(best_vars[0::3])
        
    # Reconstruct centers from optimized parameters
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
