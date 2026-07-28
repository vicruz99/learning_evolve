# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000160 (state 296f36e1) state=f1679ff0 sum of radii=2.510676 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_lp(centers, A_ub, b_ub, bounds_lp, c_obj, n, triu_idx):
    """Solves LP to maximize sum of radii for fixed centers."""
    # Update boundary constraints
    for i in range(n):
        b_ub[4*i] = centers[i, 0]
        b_ub[4*i+1] = 1.0 - centers[i, 0]
        b_ub[4*i+2] = centers[i, 1]
        b_ub[4*i+3] = 1.0 - centers[i, 1]
        
    # Update pairwise distance constraints
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub[4*n:] = dists[triu_idx]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    # 1. Initial hexagonal configuration with slight jitter to break symmetry
    r0 = 0.095
    pts = []
    y = r0
    row = 0
    while len(pts) < n:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    centers = np.array(pts[:n])
    centers += rng.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # 2. Precompute LP constraint matrix structure (constant throughout optimization)
    n_cons = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((n_cons, n))
    idx = 0
    for i in range(n):
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
        A_ub[idx, i] = 1.0; idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            idx += 1
            
    b_ub = np.zeros(n_cons)
    bounds_lp = [(0.0, None)] * n
    c_obj = -np.ones(n)
    triu_idx = np.triu_indices(n, k=1)
    
    # Initial evaluation
    best_radii, best_sum = compute_lp(centers, A_ub, b_ub, bounds_lp, c_obj, n, triu_idx)
    best_centers = centers.copy()
    current_sum = best_sum
    
    # 3. Simulated Annealing on centers
    temp = 0.006
    step = 0.03
    n_iter = 4500
    
    for it in range(n_iter):
        i = rng.integers(n)
        old_c = centers[i].copy()
        
        # Perturb one circle
        centers[i] += rng.uniform(-step, step, 2)
        centers[i] = np.clip(centers[i], 1e-4, 1.0 - 1e-4)
        
        # Evaluate new configuration
        radii, new_sum = compute_lp(centers, A_ub, b_ub, bounds_lp, c_obj, n, triu_idx)
        delta = new_sum - current_sum
        
        # Accept or reject based on SA criterion
        if delta > 0 or rng.random() < np.exp(min(delta / max(temp, 1e-6), 8.0)):
            current_sum = new_sum
            if new_sum > best_sum:
                best_sum = new_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
        else:
            centers[i] = old_c  # Revert perturbation
            
        # Cool down
        step *= 0.9995
        temp *= 0.9992
        
    # 4. Strict safety scaling to guarantee validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
