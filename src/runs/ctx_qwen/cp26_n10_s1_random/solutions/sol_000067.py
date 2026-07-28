# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000052 (state e51e4326) state=3fcdd2a7 sum of radii=2.630040 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_array, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_array[:n])

def compute_constraints(vars_array, n, triu_idx):
    """Constraints: pairwise non-overlap dist_sq >= (r_i + r_j)^2"""
    r = vars_array[:n]
    u = vars_array[n:2*n]
    v = vars_array[2*n:3*n]
    
    # Decode centers from parameterization (guarantees boundary satisfaction)
    # x = r + (1 - 2r) * u  => if u in [0,1], then x in [r, 1-r]
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    
    # Vectorized pairwise squared distances
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2
    
    # Sum of radii squared
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Return only upper triangle constraints (i < j)
    return dist_sq[triu_idx] - r_sum_sq[triu_idx]

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    
    # Bounds: r in [1e-5, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-5, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n, triu_idx)}
    
    best_vars = None
    best_sum = -np.inf
    rng = np.random.default_rng(42)
    
    def create_hex_config(r_init, seed=None):
        dy = r_init * np.sqrt(3)
        dx = 2.0 * r_init
        pts = []
        y = r_init
        row = 0
        while True:
            shift = r_init if row % 2 == 1 else 0.0
            x = r_init + shift
            while x + r_init <= 1.0:
                if len(pts) >= n: break
                pts.append([x, y])
                x += dx
            if len(pts) >= n: break
            y += dy
            row += 1
            
        pts = np.array(pts[:n])
        denom = 1.0 - 2.0 * r_init
        u = (pts[:, 0] - r_init) / denom
        v = (pts[:, 1] - r_init) / denom
        
        if seed is not None:
            rng_temp = np.random.default_rng(seed)
            u = np.clip(u + rng_temp.uniform(-0.1, 0.1, n), 0.0, 1.0)
            v = np.clip(v + rng_temp.uniform(-0.1, 0.1, n), 0.0, 1.0)
            
        return np.concatenate([np.full(n, r_init), u, v])

    # Generate diverse initial configurations
    configs = []
    configs.append(create_hex_config(0.09, seed=None)) # Exact hex
    configs.append(create_hex_config(0.09, seed=1))
    configs.append(create_hex_config(0.09, seed=2))
    configs.append(create_hex_config(0.085, seed=3))
    configs.append(create_hex_config(0.095, seed=4))
    
    # Add more randomized lattice shifts to thoroughly explore configuration space
    for seed in range(10, 30):
        cfg = create_hex_config(0.09, seed=seed)
        configs.append(cfg)

    # Optimization loop
    for x0 in configs:
        try:
            res = minimize(
                compute_objective,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                r_opt = res.x[:n]
                c_vals = compute_constraints(res.x, n, triu_idx)
                # Allow tiny numerical violation for acceptance, will fix later if needed
                if np.min(c_vals) >= -1e-6:
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    if best_vars is None:
        best_vars = configs[0]
        
    # Decode final positions
    r = best_vars[:n]
    u = best_vars[n:2*n]
    v = best_vars[2*n:3*n]
    
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    centers = np.column_stack((x, y))
    
    # Strict validity safeguard against numerical precision limits
    c_vals = compute_constraints(best_vars, n, triu_idx)
    if np.min(c_vals) < 0:
        scale = 0.99995
        for _ in range(20):
            r_scaled = r * scale
            denom_s = 1.0 - 2.0 * r_scaled
            x_s = r_scaled + denom_s * u
            y_s = r_scaled + denom_s * v
            diff_x = x_s[:, np.newaxis] - x_s[np.newaxis, :]
            diff_y = y_s[:, np.newaxis] - y_s[np.newaxis, :]
            dist_sq = diff_x**2 + diff_y**2
            r_sum_sq = (r_scaled[:, np.newaxis] + r_scaled[np.newaxis, :])**2
            if np.all(dist_sq[triu_idx] >= r_sum_sq[triu_idx] - 1e-10):
                r = r_scaled
                centers = np.column_stack((x_s, y_s))
                break
            scale -= 0.0005
            
    return centers, r, float(np.sum(r))
