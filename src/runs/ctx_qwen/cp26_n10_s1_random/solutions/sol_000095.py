# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000045 (state 7c76ac7a) state=832e42b1 sum of radii=2.439908 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_clearances(centers):
    """Compute distances to boundaries and half-pairwise distances."""
    n = centers.shape[0]
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    p = dists[dists < np.inf] / 2.0
    
    return np.concatenate([b, p])

def smooth_min_clearance(centers, k):
    """Stable log-sum-exp approximation of the minimum clearance."""
    c = compute_clearances(centers)
    max_c = np.max(c)
    exp_vals = np.exp(-k * (c - max_c))
    return -np.log(np.sum(exp_vals)) / k + max_c

def obj_neg_smooth(centers_flat, n, k):
    """Objective for L-BFGS-B: minimize negative smooth min clearance."""
    centers = centers_flat.reshape(n, 2)
    return -smooth_min_clearance(centers, k)

def obj_neg_exact_min(centers_flat, n):
    """Objective for Nelder-Mead: minimize negative exact min clearance."""
    centers = centers_flat.reshape(n, 2)
    return -np.min(compute_clearances(centers))

def get_hex_init(n, r0):
    """Generate a hexagonal lattice initialization suitable for n circles."""
    centers = []
    y = r0
    row = 0
    counts = [5, 6, 5, 6, 4]
    for cnt in counts:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(centers) < n:
                centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    return np.array(centers[:n])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    hex_init = get_hex_init(n, 0.10)
    # Normalize to comfortably fit inside [0.05, 0.95]
    hex_init = (hex_init - hex_init.min(axis=0)) / (hex_init.max(axis=0) - hex_init.min(axis=0)) * 0.9 + 0.05
    inits.append(hex_init)
    
    for _ in range(6):
        noise = rng.uniform(-0.03, 0.03, (n, 2))
        cfg = np.clip(hex_init + noise, 0.05, 0.95)
        inits.append(cfg)
        
    # Regular grid fallback
    grid_pts = []
    for i in range(6):
        for j in range(5):
            if len(grid_pts) < n:
                grid_pts.append([0.08 + j * 0.18, 0.08 + i * 0.18])
    inits.append(np.array(grid_pts[:n]))

    k = 80.0
    for cfg in inits:
        # Phase 1: Differentiable smooth optimization to find good layout
        res = minimize(obj_neg_smooth, cfg.flatten(), args=(n, k), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
        c_opt = res.x.reshape(n, 2)
        
        # Phase 2: Exact non-smooth polishing to resolve contact graph precisely
        res2 = minimize(obj_neg_exact_min, c_opt.flatten(), args=(n,), method='Nelder-Mead',
                        options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-10})
        c_final = res2.x.reshape(n, 2)
        
        # Ensure strict interior placement
        c_final = np.clip(c_final, 1e-8, 1.0 - 1e-8)
        
        # Compute exact maximum feasible radius for each circle given fixed centers
        r_exact = np.full(n, 1.0)
        for i in range(n):
            d_bound = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                          c_final[i, 1], 1.0 - c_final[i, 1])
            d_others = np.sqrt(np.sum((c_final - c_final[i])**2, axis=1))
            d_others[i] = np.inf
            r_exact[i] = min(d_bound, np.min(d_others) / 2.0)
            
        # Tiny buffer to avoid floating point edge cases during validation
        r_exact *= 0.99999
        s = np.sum(r_exact)
        
        if s > best_sum:
            best_sum = s
            best_centers = c_final.copy()
            best_radii = r_exact.copy()
            
    # Final geometric scaling to guarantee strict compliance with validate_packing tolerances
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-9: continue
        scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            r_sum = best_radii[i] + best_radii[j]
            if r_sum > 1e-9:
                scale = min(scale, dist / r_sum)
                
    best_radii *= scale * 0.99999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
