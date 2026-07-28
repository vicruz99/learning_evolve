# sol_000142 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000067 (state 3fcdd2a7) state=d65765d5 sum of radii=2.623877 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_objective(vars_array, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_array[:n])

def compute_constraints(vars_array, n, triu_idx):
    """Constraints: pairwise non-overlap dist_sq >= (r_i + r_j)^2"""
    r = vars_array[:n]
    u = vars_array[n:2*n]
    v = vars_array[2*n:3*n]
    
    # Parameterization guarantees x in [r, 1-r] and y in [r, 1-r]
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    return dist_sq[triu_idx] - r_sum_sq[triu_idx]

def generate_hex_config(row_counts, r_init, rng, perturb_std=0.0):
    """Generates an initial hexagonal lattice configuration."""
    n = 26
    centers = []
    y_curr = r_init
    for idx, count in enumerate(row_counts):
        shift = r_init if idx % 2 == 1 else 0.0
        x_start = r_init + shift
        for k in range(count):
            if len(centers) >= n: 
                break
            centers.append([x_start + k * 2.0 * r_init, y_curr])
        y_curr += r_init * np.sqrt(3)
        
    centers = np.array(centers[:n])
    
    if perturb_std > 0:
        centers += rng.normal(0, perturb_std, centers.shape)
        centers = np.clip(centers, 0.01, 0.99)
        
    denom = 1.0 - 2.0 * r_init
    u = (centers[:, 0] - r_init) / denom
    v = (centers[:, 1] - r_init) / denom
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    return np.concatenate([np.full(n, r_init), u, v])

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    
    # Max radius limited by distance to boundaries
    wall_dists = np.min([centers[:, 0], 1.0 - centers[:, 0], 
                         centers[:, 1], 1.0 - centers[:, 1]], axis=0)
    wall_dists = np.maximum(wall_dists, 0.0)
    
    c = -np.ones(n)
    bounds = [(0.0, lim) for lim in wall_dists]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    
    # Bounds: r in [1e-5, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-5, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n, triu_idx)}
    
    best_vars = None
    best_sum = -np.inf
    rng = np.random.default_rng(42)
    
    # Row patterns that sum to 26, typical for dense square packing
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [6, 6, 4, 6, 4], [5, 5, 6, 5, 5], [6, 4, 6, 4, 6],
        [6, 6, 6, 4, 4], [4, 4, 6, 6, 6], [5, 6, 6, 5, 4],
        [6, 5, 4, 6, 5], [7, 6, 7, 6], [6, 7, 6, 7]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) != 26: 
            continue
        configs.append(generate_hex_config(pat, 0.09, rng, 0.0))
        configs.append(generate_hex_config(pat, 0.09, rng, 0.02))
        configs.append(generate_hex_config(pat, 0.095, rng, 0.01))
        
    # Add fully random valid starts to explore non-lattice minima
    for _ in range(5):
        c = rng.uniform(0.2, 0.8, (n, 2))
        u = (c[:, 0] - 0.09) / (1.0 - 0.18)
        v = (c[:, 1] - 0.09) / (1.0 - 0.18)
        configs.append(np.concatenate([np.full(n, 0.09), np.clip(u, 0, 1), np.clip(v, 0, 1)]))
        
    # Phase 1: Joint optimization of centers and radii
    for x0 in configs:
        try:
            res = minimize(
                compute_objective, x0, args=(n,),
                method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False}
            )
            if np.isfinite(res.fun):
                r_opt = res.x[:n]
                c_vals = compute_constraints(res.x, n, triu_idx)
                if np.min(c_vals) >= -1e-7:
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
    best_radii = r
    
    # Phase 2: LP refinement for radii (fixes centers, optimally packs radii)
    lp_radii, lp_sum = solve_lp_radii(centers)
    if lp_radii is not None and lp_sum > best_sum:
        best_radii = lp_radii * 0.99999  # Tiny buffer for float precision
        best_sum = np.sum(best_radii)
        
    # Final strict validation & safety shrink
    wall_dists = np.min([centers[:, 0], 1.0 - centers[:, 0], 
                         centers[:, 1], 1.0 - centers[:, 1]], axis=0)
    scale = 1.0
    for _ in range(50):
        valid = True
        if np.any(best_radii > wall_dists - 1e-12):
            valid = False
        else:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            r_pair = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
            if np.any(dists[triu_idx] < r_pair[triu_idx] - 1e-12):
                valid = False
        if valid:
            break
        scale *= 0.99995
        best_radii *= scale
        
    best_sum = float(np.sum(best_radii))
    return centers, best_radii, best_sum
