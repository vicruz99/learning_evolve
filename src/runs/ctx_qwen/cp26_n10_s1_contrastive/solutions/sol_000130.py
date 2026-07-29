# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000102 (state 61650add) state=25927073 sum of radii=2.627914 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Uses parameterization to automatically satisfy boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def centers_to_params(centers, radii):
    """Convert physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    
    vars0 = np.empty(3 * N)
    vars0[0::3] = r
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def generate_force_init(seed, steps=600):
    """Generates a strictly feasible initial configuration using force-directed layout."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2)
    r_curr = np.full(N, 0.05)
    
    for step in range(steps):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-8)
        repulsion_mag = 1.0 / dists**2
        np.fill_diagonal(repulsion_mag, 0.0)
        forces = np.sum(repulsion_mag[:, :, None] * diff / dists[:, :, None], axis=1)
        
        for d in range(2):
            forces[:, d] += 5.0 * np.maximum(0, r_curr - pts[:, d])
            forces[:, d] -= 5.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
            
        step_size = 0.005 * (0.995**step)
        pts += step_size * forces
        pts = np.clip(pts, 1e-5, 1.0 - 1e-5)
        
        for i in range(N):
            d_wall = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            d_pair = np.min(np.linalg.norm(pts[i] - pts, axis=1))
            d_pair = max(d_pair, 1e-6)
            r_curr[i] = 0.8 * min(d_wall, d_pair/2.0)
            
    r_final = np.zeros(N)
    for i in range(N):
        d_wall = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        dists = np.linalg.norm(pts[i] - pts, axis=1)
        dists[i] = np.inf
        d_pair = np.min(dists)
        r_final[i] = 0.95 * min(d_wall, d_pair/2.0)
        
    return centers_to_params(pts, r_final)

def generate_hex_init(seed, rotation=0.0, scale=1.0):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.09
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
        
    centers = np.array(pts[:N])
    centers += rng.uniform(-0.02, 0.02, (N, 2))
    centers = np.clip(centers, 0.02, 0.98)
    
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        centers = (centers - 0.5) @ rot.T + 0.5
        centers = np.clip(centers, 0.02, 0.98)
        
    r_vals = np.zeros(N)
    for i in range(N):
        d_wall = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        dists = np.linalg.norm(centers[i] - centers, axis=1)
        dists[i] = np.inf
        d_pair = np.min(dists)
        r_vals[i] = 0.9 * min(d_wall, d_pair/2.0)
        
    return centers_to_params(centers, r_vals)

def run_packing():
    """
    Solves the circle packing problem for N=26 in a unit square.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -1.0
    rng = np.random.RandomState(42)
    
    inits = []
    
    # Force-based initializations (excellent at finding tight packings)
    for s in range(25):
        inits.append(generate_force_init(seed=s))
        
    # Hexagonal lattice variations with random rotations and scales
    for s in range(25):
        rot = rng.uniform(-0.3, 0.3)
        scale = rng.uniform(0.9, 1.1)
        inits.append(generate_hex_init(seed=s, rotation=rot, scale=scale))
        
    # Phase 1: Broad search from diverse initializations
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(60):
            x0 = best_vars.copy()
            pert = rng.randn(3 * N)
            # Perturb positions more aggressively to explore topology changes
            x0[1::3] += pert[1::3] * 0.03
            x0[2::3] += pert[2::3] * 0.03
            x0[0::3] += pert[0::3] * 0.002
            
            x0[0::3] = np.clip(x0[0::3], 1e-6, 0.5)
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    cons_val = constraint_func(res.x)
                    if np.min(cons_val) >= -1e-7:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res.x)) >= -1e-7:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback configuration (should rarely be reached)
    if best_vars is None:
        centers_f = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                     np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        r_f = np.full(N, 0.09)
        best_vars = centers_to_params(centers_f, r_f)
        best_sum = np.sum(r_f)

    # Reconstruct centers from optimized parameters
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
