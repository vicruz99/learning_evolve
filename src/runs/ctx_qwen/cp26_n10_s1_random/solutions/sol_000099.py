# sol_000099 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=e88dd4af sum of radii=2.253588 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    
    # Upper bounds for each radius based on distance to boundaries
    limits = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Constraint matrix: r_i + r_j <= dist(i,j)
    idx = np.triu_indices(n, k=1)
    m = len(idx[0])
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx[0]] = 1.0
    A_ub[np.arange(m), idx[1]] = 1.0
    b_ub = dists[idx]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def center_penalty(centers_flat, radii):
    """Computes a smooth penalty for boundary and overlap violations."""
    centers = centers_flat.reshape(-1, 2)
    n = len(radii)
    cx, cy = centers[:, 0], centers[:, 1]
    r = radii
    
    # Boundary penalties
    p = np.sum(np.maximum(0, r - cx)**2)
    p += np.sum(np.maximum(0, cx + r - 1.0)**2)
    p += np.sum(np.maximum(0, r - cy)**2)
    p += np.sum(np.maximum(0, cy + r - 1.0)**2)
    
    # Overlap penalties
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid self-interaction
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = np.maximum(0, r_sum - dists)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p += np.sum((viol * mask)**2)
    
    return p

def optimize_centers_for_radii(centers, radii):
    """Pushes centers apart to minimize overlap/boundary violations for fixed radii."""
    n = len(radii)
    bounds_c = [(0.0, 1.0)] * (2 * n)
    res = minimize(center_penalty, centers.flatten(), args=(radii,), 
                   method='L-BFGS-B', bounds=bounds_c, 
                   options={'maxiter': 3000, 'ftol': 1e-14})
    return res.x.reshape(-1, 2)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    starts = []
    np.random.seed(42)
    
    # 1. Base hexagonal lattice
    r_h = 0.10
    pts = []
    y = r_h
    row = 0
    while len(pts) < n and y + r_h < 1.0:
        shift = r_h if row % 2 else 0.0
        x = r_h + shift
        while x + r_h < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_h
        y += np.sqrt(3) * r_h
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    starts.append(np.array(pts[:n]))
    
    # 2. Perturbed hexagonal configurations
    for _ in range(12):
        p = starts[0].copy() + np.random.uniform(-0.035, 0.035, (n, 2))
        starts.append(np.clip(p, 0.05, 0.95))
        
    # 3. Random dense configurations
    for _ in range(6):
        starts.append(np.random.uniform(0.12, 0.88, (n, 2)))
        
    # Alternating optimization loop
    for cfg in starts:
        centers = cfg.copy()
        radii = np.full(n, 0.08)
        
        for _ in range(30):
            # Step 1: Optimize radii exactly for current centers via LP
            radii, cur_sum = solve_radii_lp(centers)
            if not np.isfinite(cur_sum):
                break
            
            # Step 2: Optimize centers to relieve pressure for current radii
            centers = optimize_centers_for_radii(centers, radii)
            
            # Check if configuration is valid/stable
            p_val = center_penalty(centers.flatten(), radii)
            if p_val < 1e-9:
                cur_sum = np.sum(radii)
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                break
        else:
            cur_sum = np.sum(radii)
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Fallback safety
    if best_centers is None:
        best_centers = starts[0]
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Final strict safety scaling to guarantee validity within 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
