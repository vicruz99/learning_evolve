# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000043 (state 8d6d3048) state=e2087257 sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_constraints(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    n = len(vars_vec) // 3
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c_list = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    # Only upper triangular pairs (i < j) to avoid duplicates and self-comparison
    i_idx, j_idx = np.triu_indices(n, k=1)
    c_list.append(dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx])
    
    return np.concatenate(c_list)

def get_objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def generate_init(n, seed, pattern='hex'):
    """Generates a strictly feasible initial configuration."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))
    
    if pattern == 'hex':
        r_est = 0.09
        y = r_est
        row = 0
        idx = 0
        while y < 1.0 - r_est + 0.01 and idx < n:
            x_start = r_est if row % 2 == 0 else 2.0 * r_est
            x = x_start
            while x < 1.0 - r_est + 0.01 and idx < n:
                centers[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
    elif pattern == 'rot_hex':
        # Generate base hex, then rotate
        pts = []
        r_est = 0.09
        y = r_est
        row = 0
        while len(pts) < n + 10:
            x_start = r_est if row % 2 == 0 else 2.0 * r_est
            x = x_start
            while x < 1.0 - r_est + 0.01:
                pts.append([x, y])
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
        pts = np.array(pts[:n+10])
        
        angle = rng.uniform(0.1, np.pi/2)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts_rot = pts @ rot.T
        
        mn = pts_rot.min(axis=0)
        mx = pts_rot.max(axis=0)
        pts_rot = (pts_rot - mn) / (mx - mn) * 0.8 + 0.1
        centers = np.clip(pts_rot[:n], 0.05, 0.95)
    elif pattern == 'grid':
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                    idx += 1
        if n > 25: centers[25] = [0.5, 0.5]
    else: # random
        centers = rng.rand(n, 2)

    # Add controlled jitter
    centers += rng.uniform(-0.015, 0.015, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)

    # Compute minimum distance to boundaries and other circles
    min_d_global = 1.0
    for i in range(n):
        d_bound = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        min_d_global = min(min_d_global, d_bound)
        for j in range(i+1, n):
            d_pair = np.linalg.norm(centers[i] - centers[j])
            min_d_global = min(min_d_global, d_pair)

    # Set initial radii to a safe fraction to guarantee strict feasibility
    r0 = min_d_global * 0.42
    
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = centers[:, 0]
    vars_init[1::3] = centers[:, 1]
    vars_init[2::3] = r0
    return vars_init

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints}

    best_vars = None
    best_sum = -np.inf

    # Phase 1: Broad search from structured and rotated initializations
    # Diverse patterns help break symmetry and explore different topological basins
    patterns = ['hex', 'hex', 'rot_hex', 'rot_hex', 'rot_hex', 'grid', 'grid', 'rand', 'rand']
    for p in patterns:
        for s in range(8):
            x0 = generate_init(n, s, p)
            try:
                res = minimize(get_objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    # Verify constraints are satisfied within numerical tolerance
                    if np.min(get_constraints(res.x)) >= -1e-7:
                        s_val = np.sum(res.x[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue

    # Phase 2: Local refinement via perturbation to escape local minima
    if best_vars is not None:
        for k in range(25):
            x0 = best_vars.copy()
            # Small Gaussian perturbation
            x0 += np.random.randn(3 * n) * 0.004
            x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
            x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
            x0[2::3] = np.clip(x0[2::3], 1e-6, 0.49)
            
            try:
                res = minimize(get_objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    if np.min(get_constraints(res.x)) >= -1e-7:
                        s_val = np.sum(res.x[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue

    # Phase 3: High-precision polish on the absolute best configuration found
    if best_vars is not None:
        try:
            res_final = minimize(get_objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_final.success and np.min(get_constraints(res_final.x)) >= -1e-7:
                best_vars = res_final.x
                best_sum = np.sum(res_final.x[2::3])
        except Exception:
            pass

    # Fallback valid configuration if optimization fails completely
    if best_vars is None:
        x0 = generate_init(n, 0, 'grid')
        res = minimize(get_objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 3000, 'disp': False})
        best_vars = res.x
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    return centers, radii, float(np.sum(radii))
