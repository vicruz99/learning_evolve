# sol_000249 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000238 (state ed5af233) state=5bceaf52 sum of radii=2.532572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def force_simulate(centers, radii, steps=4000, dt=0.004, damping=0.92, k_rep=100.0, k_wall=200.0):
    """
    Force-directed simulation that gradually expands radii while resolving overlaps.
    Returns converged centers and expanded radii.
    """
    n = len(radii)
    vel = np.zeros_like(centers)
    radii = radii.copy()
    centers = centers.copy()
    expand_rate = 8e-7 
    
    for _ in range(steps):
        # Slowly expand all circles to push them against each other
        radii *= (1.0 + expand_rate)
        
        # Compute pairwise distances
        diff_mat = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dist_mat = np.sqrt(np.sum(diff_mat**2, axis=2))
        np.fill_diagonal(dist_mat, 1e9)
        r_sum_mat = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Compute repulsive forces for overlaps
        overlap = np.maximum(0.0, r_sum_mat - dist_mat)
        force_mag = k_rep * overlap / (dist_mat + 1e-9)
        fx = np.sum(diff_mat[:, :, 0] * force_mag, axis=1)
        fy = np.sum(diff_mat[:, :, 1] * force_mag, axis=1)
        forces = np.column_stack((fx, fy))
        
        # Boundary repulsion forces
        forces[:, 0] += np.clip(radii - centers[:, 0], 0, None) * k_wall
        forces[:, 0] -= np.clip(centers[:, 0] + radii - 1.0, 0, None) * k_wall
        forces[:, 1] += np.clip(radii - centers[:, 1], 0, None) * k_wall
        forces[:, 1] -= np.clip(centers[:, 1] + radii - 1.0, 0, None) * k_wall
        
        # Update velocities and positions
        vel = damping * vel + forces * dt
        centers += vel
        centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
        
    return centers, radii

def solve_radii_lp(centers, n, A_ub, idx_i, idx_j):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Returns optimal radii and sum, or None/0 on failure.
    """
    # Maximum radius allowed by boundaries for each circle
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    wall_dists = np.maximum(wall_dists, 1e-9)
    bounds = [(0.0, lim) for lim in wall_dists]
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    try:
        # Maximize sum(r) <=> Minimize -sum(r)
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    idx_i, idx_j = triu_idx
    m = len(idx_i)
    
    # Precompute constant LP structure for pairwise constraints: r_i + r_j <= dist_ij
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    rng = np.random.default_rng(42)
    
    # Diverse hexagonal row patterns summing to 26
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 4, 6, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 4, 6],
        [7, 6, 7, 6], [6, 7, 6, 7], [5, 6, 6, 5, 4],
        [6, 5, 4, 6, 5], [5, 7, 5, 5, 4], [4, 5, 5, 5, 7]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) != n: continue
        pts = []
        y_curr = 0.09
        r_init = 0.09
        for idx, count in enumerate(pat):
            shift = r_init if idx % 2 == 1 else 0.0
            x_start = r_init + shift
            for k in range(count):
                if len(pts) >= n: break
                pts.append([x_start + k * 2.0 * r_init, y_curr])
            y_curr += r_init * np.sqrt(3)
        pts = np.array(pts[:n])
        configs.append(pts)
        
    # Add perturbed variants to break symmetry and explore landscape
    for _ in range(5):
        c = configs[0].copy()
        c += rng.uniform(-0.04, 0.04, (n, 2))
        c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    # Phase 1: Force simulation from diverse starts
    for cfg in configs:
        sim_c, sim_r = force_simulate(cfg.copy(), np.full(n, 0.085), steps=4000)
        
        # Phase 2: Exact LP refinement for radii given simulated centers
        lp_r, lp_s = solve_radii_lp(sim_c, n, A_ub, idx_i, idx_j)
        if lp_r is not None and lp_s > best_sum:
            best_sum = lp_s
            best_centers = sim_c.copy()
            best_radii = lp_r.copy()
            
    # Phase 3: LP-Guided Hill Climbing on Centers
    # Escapes local minima by directly optimizing the non-smooth LP objective
    if best_centers is not None:
        current_centers = best_centers.copy()
        current_radii = best_radii.copy()
        current_sum = best_sum
        
        for step in range(4000):
            # Adaptive decaying step size with slight baseline to avoid stalling
            step_size = 0.015 * (0.6 + 0.4 * np.exp(-step / 1500.0))
            
            # Randomly choose to move 1 or 2 circles
            n_move = rng.choice([1, 2], p=[0.85, 0.15])
            idxs = rng.choice(n, n_move, replace=False)
            old_positions = current_centers[idxs].copy()
            
            current_centers[idxs] += rng.uniform(-step_size, step_size, (n_move, 2))
            current_centers[idxs] = np.clip(current_centers[idxs], 1e-4, 1.0 - 1e-4)
            
            lp_r, lp_s = solve_radii_lp(current_centers, n, A_ub, idx_i, idx_j)
            if lp_r is not None and lp_s > current_sum + 1e-7:
                current_sum = lp_s
                current_radii = lp_r.copy()
            else:
                current_centers[idxs] = old_positions
                
        best_centers = current_centers
        best_radii = current_radii
        best_sum = current_sum

    # Fallback configuration if optimization unexpectedly fails
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict numerical safety scaling
    # Ensure all constraints strictly satisfy the 1e-12 validation tolerance
    scale = 1.0
    wall_lim = np.minimum(
        np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
        np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1])
    )
    scale = min(scale, np.min(wall_lim / np.maximum(best_radii, 1e-12)))
    
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
    scale = min(scale, np.min(dists[triu_idx] / np.maximum(r_pair[triu_idx], 1e-12)))
    
    # Apply high-precision shrinkage
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
