# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=1408359d sum of radii=2.595180 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def transform_to_centers(r, u, v):
    """Map normalized coordinates (u, v) to actual positions (x, y) ensuring boundary constraints."""
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return x, y

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraints(vars_vec):
    """Compute pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2 >= 0"""
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    x, y = transform_to_centers(r, u, v)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Extract upper triangular pairs to avoid duplicates and self-comparison
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def generate_init_hex(r_est, row_counts, perturb=0.0):
    """Generate hexagonal lattice initialization."""
    pts = []
    y = r_est
    for r_idx, count in enumerate(row_counts):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(count):
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
    pts = np.array(pts[:N])
    
    if perturb > 0:
        pts += np.random.normal(0, perturb, pts.shape)
        pts = np.clip(pts, r_est, 1.0 - r_est)
        
    r_init = np.full(N, r_est)
    denom = 1.0 - 2.0 * r_est
    u = (pts[:, 0] - r_est) / denom
    v = (pts[:, 1] - r_est) / denom
    u = np.clip(u, 0.01, 0.99)
    v = np.clip(v, 0.01, 0.99)
    
    vars0 = np.empty(N * 3)
    vars0[0::3] = r_init
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def generate_init_grid(perturb=0.0):
    """Generate 5x5 grid + center initialization."""
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    pts.append([0.5, 0.5])
    pts = np.array(pts[:N])
    
    if perturb > 0:
        pts += np.random.normal(0, perturb, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
    r_est = 0.08
    r_init = np.full(N, r_est)
    denom = 1.0 - 2.0 * r_est
    u = (pts[:, 0] - r_est) / denom
    v = (pts[:, 1] - r_est) / denom
    u = np.clip(u, 0.01, 0.99)
    v = np.clip(v, 0.01, 0.99)
    
    vars0 = np.empty(N * 3)
    vars0[0::3] = r_init
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def generate_init_random(seed, perturb=0.02):
    """Generate force-relaxed random initialization."""
    np.random.seed(seed)
    pts = np.random.rand(N, 2)
    
    # Simple repulsive force simulation to spread points
    for _ in range(300):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-5: dist = 1e-5
                f = diff / (dist**2)
                forces[i] += f
                forces[j] -= f
        pts += forces * 0.002
        pts = np.clip(pts, 0.02, 0.98)
        
    if perturb > 0:
        pts += np.random.normal(0, perturb, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        
    r_est = 0.05
    r_init = np.full(N, r_est)
    denom = 1.0 - 2.0 * r_est
    u = (pts[:, 0] - r_est) / denom
    v = (pts[:, 1] - r_est) / denom
    u = np.clip(u, 0.01, 0.99)
    v = np.clip(v, 0.01, 0.99)
    
    vars0 = np.empty(N * 3)
    vars0[0::3] = r_init
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -np.inf
    best_vars = None
    
    # Bounds: r in [1e-5, 0.49], u in [0, 1], v in [0, 1]
    bounds = [(1e-5, 0.49), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Generate diverse initial configurations
    inits = [
        generate_init_hex(0.095, [5, 6, 5, 6, 4]),
        generate_init_hex(0.095, [6, 5, 6, 5, 4]),
        generate_init_hex(0.090, [5, 6, 5, 6, 4], 0.02),
        generate_init_hex(0.090, [6, 5, 6, 5, 4], 0.02),
        generate_init_grid(0.0),
        generate_init_grid(0.02),
    ]
    for s in range(15):
        inits.append(generate_init_random(s))
        
    # Phase 1: SLSQP optimization from multiple starts
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cons_val = constraints(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(12):
            x_pert = best_vars + np.random.normal(0, 0.002, best_vars.shape)
            x_pert[0::3] = np.clip(x_pert[0::3], 1e-5, 0.49)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.0, 1.0)
            x_pert[2::3] = np.clip(x_pert[2::3], 0.0, 1.0)
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                if res.success:
                    cons_val = constraints(res.x)
                    if np.min(cons_val) >= -1e-9:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
    # Fallback to first init if optimization completely fails
    if best_vars is None:
        best_vars = inits[0]
        
    # Reconstruct physical centers from optimized parameters
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt, y_opt = transform_to_centers(r_opt, u_opt, v_opt)
    centers = np.column_stack((x_opt, y_opt))
    radii = r_opt
    
    return centers, radii, float(np.sum(radii))
