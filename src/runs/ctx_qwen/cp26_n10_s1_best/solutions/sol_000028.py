# sol_000028 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000004 (state 5455684e) state=00066a9e sum of radii=2.619404 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Compute all inequality constraints >= 0."""
    n = N_CIRCLES
    cs = np.array(x).reshape(n, 3)
    xs = cs[:, 0]
    ys = cs[:, 1]
    rs = cs[:, 2]
    
    cons = []
    
    # Boundary constraints: circle inside [0, 1] x [0, 1]
    # x - r >= 0
    cons.extend(xs - rs)
    # 1 - x - r >= 0
    cons.extend(1.0 - xs - rs)
    # y - r >= 0
    cons.extend(ys - rs)
    # 1 - y - r >= 0
    cons.extend(1.0 - ys - rs)
    
    # Overlap constraints: squared distance - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dr = rs[i] + rs[j]
            cons.append(dx*dx + dy*dy - dr*dr)
            
    return np.array(cons)

def get_init_guess(method, seed):
    """Generate initial configuration vector."""
    np.random.seed(seed)
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    
    if method == 'hex':
        # Known dense structure for N=26: 6-5-6-5-4 rows
        counts = [6, 5, 6, 5, 4]
        r0 = 0.09  # Start slightly smaller to guarantee feasibility
        dy = np.sqrt(3) * r0
        y = r0
        idx = 0
        for i, cnt in enumerate(counts):
            x_start = r0 + (0.5 * r0 if i % 2 == 1 else 0.0)
            for j in range(cnt):
                if idx < N_CIRCLES:
                    centers[idx] = [x_start + j * 2 * r0, y]
                    radii[idx] = r0
                    idx += 1
            y += dy
        # Fill remaining if any (should be exact)
        while idx < N_CIRCLES:
            centers[idx] = np.random.uniform(0.2, 0.8, 2)
            radii[idx] = 0.05
            idx += 1
    elif method == 'grid':
        r0 = 0.09
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N_CIRCLES:
                    centers[idx] = [r0 + j * 2 * r0, r0 + i * 2 * r0]
                    radii[idx] = r0
                    idx += 1
        if idx < N_CIRCLES:
            centers[idx] = [0.5, 0.5]
            radii[idx] = 0.05
            idx += 1
    else:
        # Random valid initialization
        radii[:] = 0.06
        for i in range(N_CIRCLES):
            centers[i] = np.random.uniform(radii[i], 1.0 - radii[i], 2)
            
    # Add small noise to break exact symmetry
    centers += np.random.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Flatten to [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing():
    best_sum = -1.0
    best_x = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    
    # Multiple restarts with varied seeds and initialization strategies
    seeds = list(range(30))
    methods = ['hex'] * 20 + ['grid'] * 5 + ['rand'] * 5
    
    for idx, seed in enumerate(seeds):
        method = methods[idx % len(methods)]
        x0 = get_init_guess(method, seed)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Local refinement: perturb best solution and re-optimize to escape local minima
    if best_x is not None:
        for _ in range(5):
            x_perturbed = best_x + np.random.normal(0, 1e-4, best_x.shape)
            try:
                res = minimize(objective, x_perturbed, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraints},
                               options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun) and -res.fun > best_sum:
                    best_sum = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization completely fails (unlikely)
    if best_x is None:
        best_x = get_init_guess('hex', 0)
        best_sum = -np.sum(best_x[2::3])
        
    # Extract and format results
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    # Ensure non-negative radii (safety against numerical drift)
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(best_sum)
