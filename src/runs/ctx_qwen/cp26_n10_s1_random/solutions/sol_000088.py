# sol_000088 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=2891ee63 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_flat, n):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(vars_flat[2::3])

def get_constraints(vars_flat, n):
    """Computes inequality constraints >= 0 for valid packing."""
    x = vars_flat[0::3]
    y = vars_flat[1::3]
    r = vars_flat[2::3]
    
    c_list = []
    
    # Boundary constraints
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    rows, cols = np.triu_indices(n, k=1)
    c_list.append(dists[rows, cols] - r_sum[rows, cols])
    
    return np.concatenate(c_list)

def generate_hex_config(n, row_counts, r_init):
    """Generates a hexagonal lattice configuration based on row counts."""
    centers = []
    y = r_init
    for idx, count in enumerate(row_counts):
        shift = r_init if idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(count):
            if len(centers) >= n:
                break
            centers.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
    return np.array(centers[:n])

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.05, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n,)}
    
    np.random.seed(42)
    configs = []
    
    # 1. Hexagonal patterns with various row distributions
    row_distributions = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [5, 6, 6, 5, 4]
    ]
    
    for dist in row_distributions:
        hex_cfg = generate_hex_config(n, dist, 0.085)
        # Normalize to center in square
        hex_cfg = (hex_cfg - hex_cfg.min(axis=0)) / (hex_cfg.max(axis=0) - hex_cfg.min(axis=0))
        hex_cfg = hex_cfg * 0.85 + 0.075
        
        # Add base config
        v0 = np.zeros(3 * n)
        v0[0::3] = hex_cfg[:, 0]
        v0[1::3] = hex_cfg[:, 1]
        v0[2::3] = 0.085
        configs.append(v0)
        
        # Add perturbations
        for _ in range(3):
            v_pert = v0.copy()
            noise = np.random.uniform(-0.02, 0.02, (n, 2))
            p = np.clip(hex_cfg + noise, 0.05, 0.95)
            v_pert[0::3] = p[:, 0]
            v_pert[1::3] = p[:, 1]
            v_pert[2::3] = 0.085 + np.random.uniform(-0.005, 0.005)
            configs.append(v_pert)
            
    # 2. Grid + center initialization
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    grid_pts.append([0.5, 0.5])
    grid_cfg = np.array(grid_pts[:n])
    
    v_grid = np.zeros(3 * n)
    v_grid[0::3] = grid_cfg[:, 0]
    v_grid[1::3] = grid_cfg[:, 1]
    v_grid[2::3] = 0.08
    configs.append(v_grid)
    
    # Add perturbations to grid
    for _ in range(4):
        v_p = v_grid.copy()
        noise = np.random.uniform(-0.025, 0.025, (n, 2))
        p = np.clip(grid_cfg + noise, 0.05, 0.95)
        v_p[0::3] = p[:, 0]
        v_p[1::3] = p[:, 1]
        v_p[2::3] = 0.08 + np.random.uniform(-0.005, 0.005)
        configs.append(v_p)

    # Optimize from each configuration
    for cfg in configs:
        try:
            res = minimize(
                objective,
                cfg,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < -2.4:
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                
                # Strict validity check
                valid = True
                if np.any(x_opt - r_opt < -1e-10) or np.any(x_opt + r_opt > 1 + 1e-10): valid = False
                if np.any(y_opt - r_opt < -1e-10) or np.any(y_opt + r_opt > 1 + 1e-10): valid = False
                
                if valid:
                    centers = np.column_stack((x_opt, y_opt))
                    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                    r_sums = r_opt[:, np.newaxis] + r_opt[np.newaxis, :]
                    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                    if np.any(dists[mask] < r_sums[mask] - 1e-10):
                        valid = False
                
                if valid:
                    current_sum = np.sum(r_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = centers.copy()
                        best_radii = r_opt.copy()
        except Exception:
            continue

    # Fallback if optimization yields nothing valid
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.08)
        k = 0
        for i in range(6):
            for j in range(5):
                if k >= n: break
                best_centers[k] = [0.1 + j * 0.18, 0.1 + i * 0.18]
                k += 1
        best_sum = np.sum(best_radii)

    # Safety scaling to guarantee strict validity against 1e-12 tolerance
    # Only scale if necessary, otherwise keep maximal radii
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-9: continue
        scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            r_sum = best_radii[i] + best_radii[j]
            if r_sum > 1e-9:
                scale = min(scale, dist / r_sum)
                
    if scale < 0.999999:
        best_radii *= scale * 0.99999
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
