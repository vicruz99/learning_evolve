# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state abc5794a) state=5778b268 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0

    # vars layout: [x1, y1, r1, x2, y2, r2, ...]
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)

    def constraint_vector(vars):
        # Extract x, y, r from the interleaved variables array
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        c_list = []
        # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        c_list.append(x - r)
        c_list.append(1.0 - x - r)
        c_list.append(y - r)
        c_list.append(1.0 - y - r)
        
        # Overlap constraints: dist >= r_i + r_j
        # Compute distance matrix
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dists = np.sqrt(x_diff**2 + y_diff**2)
        
        # Compute sum of radii matrix
        r_sum = r[:, None] + r[None, :]
        
        # We only need constraints for i < j (upper triangle excluding diagonal)
        rows, cols = np.triu_indices(n, k=1)
        c_list.append(dists[rows, cols] - r_sum[rows, cols])
        
        return np.concatenate(c_list)

    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    # Layout is [x1, y1, r1, x2, y2, r2, ...], so we repeat the bound triplet n times
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    configs = []
    
    # 1. Random starts to explore solution space
    for _ in range(3):
        v = np.zeros(3 * n)
        v[0::3] = np.random.rand(n)
        v[1::3] = np.random.rand(n)
        v[2::3] = 0.02 * np.ones(n) # Small initial radius to ensure feasibility
        configs.append(v)

    # 2. Grid based initialization (5x5 grid + 1)
    v = np.zeros(3 * n)
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                v[3*idx] = 0.1 + i * 0.2
                v[3*idx+1] = 0.1 + j * 0.2
                v[3*idx+2] = 0.08
                idx += 1
    if idx < n:
        v[3*idx] = 0.5
        v[3*idx+1] = 0.5
        v[3*idx+2] = 0.05
    configs.append(v)

    # 3. Hexagonal based initialization (denser packing)
    v = np.zeros(3 * n)
    idx = 0
    y = 0.1
    shift = 0
    while idx < n:
        x = 0.1 + shift * 0.1
        while x <= 0.9 and idx < n:
            v[3*idx] = x
            v[3*idx+1] = y
            v[3*idx+2] = 0.08
            idx += 1
            x += 0.2
        y += 0.1732 # sqrt(3)/2 * 0.2 approx
        shift = 1 - shift
    configs.append(v)

    constraint_dict = {'type': 'ineq', 'fun': constraint_vector}

    for cfg in configs:
        try:
            res = opt.minimize(
                objective,
                cfg,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_dict,
                options={'maxiter': 3000, 'ftol': 1e-10}
            )
            
            if res.success:
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                
                # Validation check
                valid = True
                if np.any(r_opt < -1e-9): valid = False
                if np.any(x_opt - r_opt < -1e-9) or np.any(x_opt + r_opt > 1 + 1e-9): valid = False
                if np.any(y_opt - r_opt < -1e-9) or np.any(y_opt + r_opt > 1 + 1e-9): valid = False
                
                if valid:
                    centers = np.column_stack((x_opt, y_opt))
                    # Check overlaps
                    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
                    r_sums = r_opt[:, None] + r_opt[None, :]
                    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                    if np.any(dists[mask] < r_sums[mask] - 1e-9):
                        valid = False
                
                if valid:
                    current_sum = np.sum(r_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = centers
                        best_radii = r_opt
        except Exception:
            continue

    # Fallback if optimization didn't find a good solution
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        k = 0
        # Simple grid packing
        for i in range(6):
            for j in range(5):
                if k >= n: break
                best_centers[k] = [0.1 + j*0.18, 0.1 + i*0.18]
                best_radii[k] = 0.08
                k += 1
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
