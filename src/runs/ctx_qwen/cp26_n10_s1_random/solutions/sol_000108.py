# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=c7c765f4 sum of radii=1.940666 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def min_clearance(centers):
    """Computes the minimum distance from any circle center to boundaries or other centers."""
    # Distance to square boundaries
    d_boundary = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    min_b = np.min(d_boundary)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_p = np.min(dists)
    
    return min(min_b, min_p)

def neg_min_clearance(x_flat):
    """Objective function for optimizer: maximize min clearance."""
    return -min_clearance(x_flat.reshape(N_CIRCLES, 2))

def force_sim(seed, steps=3000):
    """Generates a dense packing configuration using repulsive forces."""
    np.random.seed(seed)
    centers = np.random.uniform(0.2, 0.8, (N_CIRCLES, 2))
    r_target = 0.09
    
    for step in range(steps):
        # Gradually increase target radius to force tighter packing
        r_target = 0.09 + 0.015 * (step / steps)
        
        # Pairwise repulsion
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlap = np.maximum(0, 2.0 * r_target - dists)
        force_mag = overlap * 50.0
        inv_dists = np.where(dists > 1e-6, 1.0 / dists, 0.0)
        dirs = diff * inv_dists[:, :, None]
        forces = np.sum(dirs * force_mag[:, :, None], axis=1)
        
        # Boundary repulsion (vectorized)
        forces[:, 0] += np.where(centers[:, 0] < r_target, (r_target - centers[:, 0]) * 100.0, 0.0)
        forces[:, 0] -= np.where(centers[:, 0] > 1.0 - r_target, (centers[:, 0] - (1.0 - r_target)) * 100.0, 0.0)
        forces[:, 1] += np.where(centers[:, 1] < r_target, (r_target - centers[:, 1]) * 100.0, 0.0)
        forces[:, 1] -= np.where(centers[:, 1] > 1.0 - r_target, (centers[:, 1] - (1.0 - r_target)) * 100.0, 0.0)
            
        centers += forces * 0.005
        centers = np.clip(centers, 0.0, 1.0)
        
    return centers

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Force simulations from multiple seeds
    for s in range(10):
        configs.append(force_sim(s))
        
    # 2. Hexagonal lattice patterns (known to be near-optimal for dense packings)
    for r_h in [0.09, 0.095, 0.10]:
        pts = []
        row = 0
        y = r_h
        counts = [5, 6, 5, 6, 4]  # Sums to 26
        for cnt in counts:
            shift = r_h if row % 2 == 1 else 0.0
            x_start = r_h + shift
            for k in range(cnt):
                if len(pts) >= N_CIRCLES: break
                pts.append([x_start + k * 2 * r_h, y])
            y += r_h * np.sqrt(3)
            row += 1
        if len(pts) == N_CIRCLES:
            pts = np.array(pts)
            # Normalize to fit comfortably within [0.1, 0.9]
            pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))
            pts = pts * 0.8 + 0.1
            configs.append(pts)

    # Optimize each configuration
    for cfg in configs:
        # Polish centers to maximize minimum clearance
        res = minimize(neg_min_clearance, cfg.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9})
        
        opt_centers = res.x.reshape(N_CIRCLES, 2)
        # Ensure strict interior placement for numerical stability
        opt_centers = np.clip(opt_centers, 1e-4, 1.0 - 1e-4)
        
        # Solve LP to maximize sum of radii for fixed optimal centers
        limits = np.minimum(np.minimum(opt_centers[:, 0], 1.0 - opt_centers[:, 0]),
                            np.minimum(opt_centers[:, 1], 1.0 - opt_centers[:, 1]))
        limits = np.maximum(limits, 0.0)
        
        diffs = opt_centers[:, np.newaxis, :] - opt_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        m = N_CIRCLES * (N_CIRCLES - 1) // 2
        A_ub = np.zeros((m + 4 * N_CIRCLES, N_CIRCLES))
        b_ub = np.zeros(m + 4 * N_CIRCLES)
        
        idx = 0
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                A_ub[idx, i] = 1.0
                A_ub[idx, j] = 1.0
                b_ub[idx] = dists[i, j]
                idx += 1
                
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
        for i in range(N_CIRCLES):
            A_ub[idx, i] = 1.0; b_ub[idx] = opt_centers[i, 0]; idx += 1
            A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - opt_centers[i, 0]; idx += 1
            A_ub[idx, i] = 1.0; b_ub[idx] = opt_centers[i, 1]; idx += 1
            A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - opt_centers[i, 1]; idx += 1
            
        # Maximize sum(r) -> Minimize -sum(r)
        lp_res = linprog(-np.ones(N_CIRCLES), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            current_sum = -lp_res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = opt_centers.copy()
                best_radii = lp_res.x.copy()
                
    # Apply tiny safety margin to strictly satisfy 1e-12 validation tolerance
    best_radii *= 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
