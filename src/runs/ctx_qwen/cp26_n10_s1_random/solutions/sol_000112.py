# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=caa8e658 sum of radii=2.066312 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_min_dist(centers):
    """Computes the minimum pairwise distance and distance to boundaries."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    dists_bdry = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    min_bdry = np.min(dists_bdry)
    
    return min(min_pair, min_bdry)

def objective_min_dist(c_flat):
    """Objective to maximize min distance by minimizing its negative."""
    return -compute_min_dist(c_flat.reshape(N, 2))

def force_relax(init_centers, seed=0, steps=1000):
    """Force-directed relaxation to spread circles and find a dense valid packing."""
    centers = init_centers.copy()
    r_target_init = 0.085
    r_target_final = 0.104
    
    for step in range(steps):
        r_target = r_target_init + (r_target_final - r_target_init) * (step / steps)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        min_dists = 2.0 * r_target
        overlaps = np.maximum(0, min_dists - dists)
        mask = dists > 1e-9
        inv_dists = np.where(mask, 1.0 / dists, 0.0)
        forces = np.sum(overlaps[:, :, np.newaxis] * diff * inv_dists[:, :, np.newaxis], axis=1)
        
        for dim in range(2):
            dist_l = centers[:, dim] - r_target
            forces[:, dim] += np.maximum(0, -dist_l) * 5.0
            dist_r = (1.0 - centers[:, dim]) - r_target
            forces[:, dim] += np.maximum(0, -dist_r) * 5.0
            
        dt = 0.01 / (1.0 + step/50)
        centers += forces * dt
        centers = np.clip(centers, r_target, 1.0 - r_target)
        
    return centers

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m + n, n))
    b_ub = np.zeros(m + n)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(n):
        A_ub[idx, i] = 1.0
        b_ub[idx] = limits[i]
        idx += 1
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing():
    np.random.seed(42)
    configs = []
    
    # 1. Hexagonal-like row distribution: 6, 5, 6, 5, 4
    cfg = []
    y = 0.1
    rows = [6, 5, 6, 5, 4]
    for i, cnt in enumerate(rows):
        shift = 0.05 if i % 2 == 1 else 0.0
        x_start = 0.1 + shift
        if cnt > 1:
            xs = np.linspace(x_start, 0.9, cnt)
        else:
            xs = [0.5]
        for x in xs:
            cfg.append([x, y])
        y += 0.19
    configs.append(np.array(cfg[:N]))
    
    # 2. Dense grid 5x5 + 1
    g = np.linspace(0.12, 0.88, 5)
    cfg2 = np.array([[x, y] for y in g for x in g])
    cfg2 = np.vstack([cfg2, [[0.5, 0.5]]])
    configs.append(cfg2)
    
    # 3-9. Perturbed variants to ensure diverse basins of attraction
    for i in range(7):
        base = configs[i % 2]
        p = base + np.random.uniform(-0.04, 0.04, (N, 2))
        configs.append(np.clip(p, 0.05, 0.95))
        
    # Relax each configuration
    relaxed = []
    for i, cfg in enumerate(configs):
        relaxed.append(force_relax(cfg, seed=i*13))
        
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Optimize centers and solve LP for radii
    for cfg in relaxed:
        res = minimize(objective_min_dist, cfg.flatten(), method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
        opt_c = res.x.reshape(N, 2)
        radii, s = solve_lp_radii(opt_c)
        if radii is not None and s > best_sum:
            best_sum = s
            best_centers = opt_c.copy()
            best_radii = radii.copy()
            
    # Fallback if LP fails (highly unlikely with this setup)
    if best_radii is None:
        best_centers = configs[0]
        best_radii = np.full(N, 0.09)
        best_sum = np.sum(best_radii)
        
    # Apply tiny shrink to strictly satisfy 1e-12 validation tolerance
    best_radii *= 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
