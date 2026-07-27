# sol_000203 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a6b07ba) state=3e7a5411 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = np.zeros((n_circles, 2))
    best_radii = np.zeros(n_circles)

    # Helper function for constraints
    def get_constraints(centers, radii):
        cons = []
        n = len(radii)
        
        # Boundary constraints: center - r >= 0 and center + r <= 1
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda c, r, idx=i, dim=0: c[idx, dim] - r[idx]})
            cons.append({'type': 'ineq', 'fun': lambda c, r, idx=i, dim=0: 1 - c[idx, dim] - r[idx]})
            cons.append({'type': 'ineq', 'fun': lambda c, r, idx=i, dim=1: c[idx, dim] - r[idx]})
            cons.append({'type': 'ineq', 'fun': lambda c, r, idx=i, dim=1: 1 - c[idx, dim] - r[idx]})
            
        # Non-overlap constraints: dist >= r1 + r2
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda c, r, i=i, j=j: np.linalg.norm(c[i] - c[j]) - (r[i] + r[j])
                })
        return cons

    def objective(x, n_circles):
        radii = x[n_circles * 2:]
        return -np.sum(radii)

    def get_bounds(n_circles):
        b = []
        for _ in range(n_circles):
            b.extend([(0, 1), (0, 1)]) # centers
        for _ in range(n_circles):
            b.append((0, 0.5)) # radii
        return b

    def optimize_from_start(start_centers, start_radii, n_circles):
        x0 = np.concatenate([start_centers.flatten(), start_radii])
        bounds = get_bounds(n_circles)
        cons = get_constraints(start_centers, start_radii)
        
        try:
            res = minimize(objective, x0, args=(n_circles,), method='SLSQP', 
                           bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success and res.fun < -best_sum_radii:
                return res.x, -res.fun
        except Exception:
            pass
        return None, 0.0

    # 1. Square Grid with perturbation
    c_sq = np.array([[0.1 + 0.2*i + 0.01*np.random.randn(), 0.1 + 0.2*j + 0.01*np.random.randn()] 
                     for i in range(5) for j in range(5)])[:26]
    r_sq = np.full(26, 0.1)
    
    x, s = optimize_from_start(c_sq, r_sq, n_circles)
    if x is not None and s > best_sum_radii:
        best_sum_radii = s
        best_centers = x[:52].reshape(n_circles, 2)
        best_radii = x[52:]

    # 2. Hexagonal Grid
    hex_centers = []
    r_est = 0.09
    row_idx = 0
    while len(hex_centers) < 26:
        y = (row_idx + 1) * r_est * np.sqrt(3)
        offset = (r_est / 2) if row_idx % 2 == 1 else 0
        x_val = offset + r_est
        while x_val <= 1 - r_est and len(hex_centers) < 26:
            if len(hex_centers) < 26:
                hex_centers.append([x_val, y])
                hex_centers[-1][0] += 0.01 * np.random.randn()
                hex_centers[-1][1] += 0.01 * np.random.randn()
            x_val += 2 * r_est
        row_idx += 1
    c_hex = np.array(hex_centers[:26])
    r_hex = np.full(26, r_est)
    
    x, s = optimize_from_start(c_hex, r_hex, n_circles)
    if x is not None and s > best_sum_radii:
        best_sum_radii = s
        best_centers = x[:52].reshape(n_circles, 2)
        best_radii = x[52:]

    # 3. Random initialization
    for _ in range(5):
        c_rand = np.random.rand(n_circles, 2)
        r_rand = np.random.uniform(0.05, 0.1, n_circles)
        x, s = optimize_from_start(c_rand, r_rand, n_circles)
        if x is not None and s > best_sum_radii:
            best_sum_radii = s
            best_centers = x[:52].reshape(n_circles, 2)
            best_radii = x[52:]

    return best_centers, best_radii, best_sum_radii
