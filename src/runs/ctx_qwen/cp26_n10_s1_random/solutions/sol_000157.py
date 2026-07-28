# sol_000157 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000140 (state 07ed95ff) state=4a0dd77a sum of radii=2.250000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_clearance(centers):
    """Computes the maximum feasible equal radius for given centers."""
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundaries
    min_wall = np.min(np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    ))
    
    return min(min_pair, min_wall)

def neg_clearance(x):
    """Objective for optimizer: maximize clearance => minimize negative clearance."""
    centers = x.reshape(-1, 2)
    # Penalize heavily if outside bounds to keep search feasible
    if np.any(centers < 0) or np.any(centers > 1.0):
        return -100.0
    return -compute_clearance(centers)

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    
    # Upper bounds from boundaries
    limits = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    bounds = [(0.0, max(0.0, lim)) for lim in limits]
    
    # Pairwise distance constraints: r_i + r_j <= dist(i,j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diffs = centers[idx_i] - centers[idx_j]
    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
    
    c_obj = -np.ones(n)  # Maximize sum => minimize negative sum
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def generate_hex_config(r0, seed_val):
    """Generates a hexagonal lattice configuration."""
    rng = np.random.default_rng(seed_val)
    pts = []
    y = r0
    row = 0
    # Generate enough points to fill 26
    while len(pts) < 35 and y + r0 <= 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts) < 35:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    # If lattice didn't yield enough, pad near center
    while len(pts) < 26:
        pts.append([0.5 + rng.uniform(-0.15, 0.15), 
                    0.5 + rng.uniform(-0.15, 0.15)])
                    
    return np.array(pts[:26])

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    rng = np.random.default_rng(42)
    
    configs = []
    
    # 1. Hexagonal lattices with varying base radii
    for r0 in np.linspace(0.07, 0.115, 10):
        configs.append(generate_hex_config(r0, rng.integers(10000)))
        for _ in range(2):
            cfg = generate_hex_config(r0, rng.integers(10000))
            cfg += rng.uniform(-0.025, 0.025, (n, 2))
            configs.append(np.clip(cfg, 0.05, 0.95))
            
    # 2. Perturbed regular grids
    for offset in [0.0, 0.05, 0.1]:
        grid = np.array([(offset + i*0.2, offset + j*0.2) for j in range(5) for i in range(5)])
        grid = np.vstack([grid, [0.5, 0.5]])
        cfg = grid + rng.uniform(-0.03, 0.03, (n, 2))
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # 3. Random dense configurations
    for _ in range(5):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Phase 1: Optimize centers to maximize clearance
    for cfg in configs:
        try:
            res = minimize(neg_clearance, cfg.flatten(), method='Powell',
                           options={'maxiter': 30000, 'xtol': 1e-9, 'ftol': 1e-14})
            c_opt = res.x.reshape(n, 2)
            c_opt = np.clip(c_opt, 1e-9, 1.0 - 1e-9)
            
            # Phase 2: Exact LP for variable radii
            r_lp, s_lp = solve_lp_radii(c_opt)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_opt.copy()
                best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 3: Local perturbation refinement around best configuration
    if best_centers is not None:
        for _ in range(80):
            c_pert = best_centers + rng.normal(0, 0.002, (n, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_lp, s_lp = solve_lp_radii(c_pert)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_pert.copy()
                best_radii = r_lp.copy()
                
    # Apply strict safety margin to guarantee numerical validity
    if best_radii is not None:
        best_radii *= 0.99999995
        best_sum = float(np.sum(best_radii))
        
    # Fallback (should not be reached given robust optimization)
    if best_centers is None:
        best_centers = generate_hex_config(0.09, 42)
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
