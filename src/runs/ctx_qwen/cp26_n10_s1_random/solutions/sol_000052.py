# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state 6f2d6856) state=e51e4326 sum of radii=2.618042 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_array, n, triu_idx):
    """Minimize negative sum of radii"""
    return -np.sum(vars_array[:n])

def constraint_fun(vars_array, n, triu_idx):
    """
    Computes pairwise non-overlap constraints:
    dist_sq(i,j) - (r_i + r_j)^2 >= 0
    """
    r = vars_array[:n]
    u = vars_array[n:2*n]
    v = vars_array[2*n:3*n]
    
    # Decode centers from parameterization (guarantees boundary satisfaction)
    x = r + (1.0 - 2.0*r)*u
    y = r + (1.0 - 2.0*r)*v
    
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
    # Precompute indices for upper triangle (i < j)
    triu_idx = np.triu_indices(n, k=1)
    
    # Bounds: r in [1e-4, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-4, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    
    cons = {'type': 'ineq', 'fun': constraint_fun, 'args': (n, triu_idx)}
    
    best_vars = None
    best_sum = -np.inf
    
    configs = []
    
    # Helper to create hex grid initialization
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
            np.random.seed(seed)
            u = np.clip(u + np.random.uniform(-0.05, 0.05, n), 0.0, 1.0)
            v = np.clip(v + np.random.uniform(-0.05, 0.05, n), 0.0, 1.0)
            
        return np.concatenate([np.full(n, r_init), u, v])

    # Configuration 1: Strict hex grid (feasible start)
    configs.append(create_hex_config(0.085))
    
    # Configuration 2-4: Perturbed hex grids to escape local minima
    configs.append(create_hex_config(0.085, seed=42))
    configs.append(create_hex_config(0.085, seed=123))
    configs.append(create_hex_config(0.085, seed=999))
    
    # Configuration 5: Uniform grid alternative
    grid_x = np.linspace(0.12, 0.88, 6)
    grid_y = np.linspace(0.12, 0.88, 5)
    grid_pts = []
    for gy in grid_y:
        for gx in grid_x:
            if len(grid_pts) >= n: break
            grid_pts.append([gx, gy])
    grid_pts = np.array(grid_pts[:n])
    r_grid = 0.085
    u_grid = (grid_pts[:, 0] - r_grid) / (1.0 - 2.0*r_grid)
    v_grid = (grid_pts[:, 1] - r_grid) / (1.0 - 2.0*r_grid)
    configs.append(np.concatenate([np.full(n, r_grid), u_grid, v_grid]))
    
    # Optimization loop
    for i, x0 in enumerate(configs):
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                args=(n, triu_idx),
                options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                r_opt = res.x[:n]
                s = np.sum(r_opt)
                # Verify constraint satisfaction
                c_vals = constraint_fun(res.x, n, triu_idx)
                if np.min(c_vals) >= -1e-6:
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
    
    x = r + (1.0 - 2.0*r)*u
    y = r + (1.0 - 2.0*r)*v
    centers = np.column_stack((x, y))
    
    # Strict validity safeguard against numerical precision limits
    c_vals = constraint_fun(best_vars, n, triu_idx)
    if np.min(c_vals) < 0:
        # Negligible scale down to guarantee dist >= r_i + r_j - 1e-12
        r *= 0.99999
        x = r + (1.0 - 2.0*r)*u
        y = r + (1.0 - 2.0*r)*v
        centers = np.column_stack((x, y))
        
    return centers, r, float(np.sum(r))
