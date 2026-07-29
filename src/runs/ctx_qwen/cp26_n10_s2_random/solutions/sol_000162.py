# sol_000162 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000133 (state 27fd9551) state=bb2a392c sum of radii=2.302853 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

N_CIRCLES = 26

def solve_lp_and_grad(centers):
    """
    Solves LP for optimal radii given fixed centers and computes the exact gradient
    of the sum of radii with respect to center positions using LP dual variables.
    """
    c = centers
    n = c.shape[0]
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    pair_idx = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            pair_idx.append((i, j))
            idx += 1
            
    c_obj = -np.ones(n)
    ub = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                    np.minimum(c[:, 1], 1.0 - c[:, 1]))
    ub = np.maximum(ub, 1e-9)
    bounds_r = [(0.0, u) for u in ub]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if not res.success:
        return np.zeros(n), 0.0, np.zeros_like(c)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Compute gradient using dual marginals
    # For active constraint r_i + r_j <= dist_ij, the dual mu_ij > 0 indicates
    # that increasing the distance allows larger radii.
    grad = np.zeros_like(c)
    marginals = res.ineqlin.marginals
    idx = 0
    for i, j in pair_idx:
        mu = marginals[idx]
        if mu > 1e-6:
            d = dists[i, j]
            if d > 1e-9:
                vec = (c[i] - c[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        idx += 1
        
    return radii, s_sum, grad

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # --- Phase 1: Generate Diverse Initial Configurations ---
    starts = []
    
    # 1. Hexagonal lattice patterns (promote high density)
    patterns = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [6,6,5,5,4], [4,6,6,6,4], [5,5,6,5,5]]
    for pat in patterns:
        c = []
        r_est = 0.098
        y = r_est
        for r_idx, cnt in enumerate(pat):
            shift = r_est if r_idx % 2 == 1 else 0.0
            x = r_est + shift
            for _ in range(cnt):
                if len(c) < N_CIRCLES:
                    c.append([x, y])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3)
        starts.append(np.array(c[:N_CIRCLES]))
        
    # 2. Force-repulsion spread configurations (avoid clustering)
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N_CIRCLES, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist = np.maximum(dist, 1e-4)
            f = np.sum(diff / (dist**2)[:, :, None], axis=1)
            c += 0.003 * f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # --- Phase 2: LP-Gradient Ascent Optimization ---
    for c0 in starts:
        c = c0.copy()
        curr_best_sum = -1.0
        
        step = 0.006
        for k in range(1500):
            radii, s, grad = solve_lp_and_grad(c)
            
            if s > curr_best_sum:
                curr_best_sum = s
                if s > best_sum:
                    best_sum = s
                    best_c = c.copy()
                    best_r = radii.copy()
                    
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-9:
                break
                
            # Step along normalized gradient direction
            c += step * (grad / g_norm)
            c = np.clip(c, 0.02, 0.98)
            
            # Adaptive step decay
            if k % 150 == 0 and k > 0:
                step *= 0.7
                
            # Periodic jitter to escape local minima
            if k % 400 == 200:
                c += rng.normal(0, 0.005, c.shape)
                c = np.clip(c, 0.03, 0.97)
                
    # --- Phase 3: Local Refinement from Best Configuration ---
    if best_c is not None:
        for _ in range(5):
            c_local = best_c + rng.normal(0, 0.002, best_c.shape)
            c_local = np.clip(c_local, 0.05, 0.95)
            step = 0.003
            for k in range(500):
                radii, s, grad = solve_lp_and_grad(c_local)
                if s > best_sum:
                    best_sum = s
                    best_c = c_local.copy()
                    best_r = radii.copy()
                g_norm = np.linalg.norm(grad)
                if g_norm < 1e-9:
                    break
                c_local += step * (grad / g_norm)
                c_local = np.clip(c_local, 0.02, 0.98)
                if k % 100 == 0:
                    step *= 0.8

    # Final exact LP solve for the best centers found
    final_r, final_s, _ = solve_lp_and_grad(best_c)
    best_r = final_r
    best_sum = final_s
    
    # --- Phase 4: Strict Numerical Repair ---
    centers = best_c.copy()
    radii = best_r.copy()
    
    for _ in range(100):
        changed = False
        
        # Resolve overlaps by symmetric shrinking
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-9:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Clamp to boundaries
        for i in range(N_CIRCLES):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            mr = min(x, 1.0 - x, y, 1.0 - y)
            if r > mr + 1e-9:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
