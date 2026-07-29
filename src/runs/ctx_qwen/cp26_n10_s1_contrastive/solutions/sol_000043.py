# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=8d6d3048 sum of radii=2.624143 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Computes inequality constraints g(x) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    n = len(x) // 3
    C = x.reshape(n, 3)
    xc = C[:, 0]
    yc = C[:, 1]
    r = C[:, 2]

    c_list = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_list.append(xc - r)
    c_list.append(1.0 - xc - r)
    c_list.append(yc - r)
    c_list.append(1.0 - yc - r)

    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2

    # Only upper triangular pairs (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    c_list.append(dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx])

    return np.concatenate(c_list)

def get_initialization(n, seed, pattern):
    """Generates a strictly feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    
    if pattern == 'hex':
        r_est = 0.095
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
    elif pattern == 'grid':
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        if n > 25:
            centers[25] = [0.5, 0.5]
    else: # random
        centers = np.random.rand(n, 2)

    # Add controlled jitter
    centers += np.random.uniform(-0.015, 0.015, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)

    # Compute minimum distance to boundaries and other circles
    min_d = 1.0
    for i in range(n):
        d_bound = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        min_d = min(min_d, d_bound)
        for j in range(i+1, n):
            d_pair = np.linalg.norm(centers[i] - centers[j])
            min_d = min(min_d, d_pair)

    # Set initial radii to a safe fraction to guarantee strict feasibility
    r0 = min_d * 0.38
    
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = centers[:, 0]
    vars_init[1::3] = centers[:, 1]
    vars_init[2::3] = r0
    return vars_init

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}

    best_vars = None
    best_sum = -np.inf

    # Phase 1: Broad search from structured initializations
    patterns = ['hex', 'hex', 'grid', 'grid', 'rand', 'rand']
    for p in patterns:
        for s in range(4):
            x0 = get_initialization(n, s, p)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-11})
                if res.success:
                    # Verify constraints are satisfied within numerical tolerance
                    if np.min(constraints(res.x)) >= -1e-6:
                        s_val = np.sum(res.x[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue

    # Phase 2: Local refinement via perturbation
    if best_vars is not None:
        for k in range(10):
            x0 = best_vars.copy()
            # Small random perturbation to escape local minima
            x0 += np.random.uniform(-0.004, 0.004, x0.shape)
            x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
            x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
            x0[2::3] = np.clip(x0[2::3], 1e-6, 0.49)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
                if res.success:
                    if np.min(constraints(res.x)) >= -1e-6:
                        s_val = np.sum(res.x[2::3])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue

    if best_vars is not None:
        centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
        radii = best_vars[2::3]
        return centers, radii, float(np.sum(radii))

    # Fallback valid configuration
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
            idx += 1
    centers[25] = [0.5, 0.5]
    radii = np.full(n, 0.09)
    return centers, radii, float(np.sum(radii))
