# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state 6f2d6856) state=7a002c32 sum of radii=2.472627 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Distance to nearest boundary for each circle
    limits = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    
    c = np.ones(n) * -1  # Maximize sum(r) -> minimize -sum(r)
    bounds = [(0, lim) for lim in limits]
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Constraint matrix: r_i + r_j <= dist(i, j)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def penalty_obj(params, n, mu, r_target):
    """Penalty objective for center optimization."""
    centers = params.reshape(n, 2)
    p = 0.0
    
    # Boundary penalties
    p += np.sum(np.maximum(0, r_target - centers[:, 0])**2)
    p += np.sum(np.maximum(0, centers[:, 0] + r_target - 1.0)**2)
    p += np.sum(np.maximum(0, r_target - centers[:, 1])**2)
    p += np.sum(np.maximum(0, centers[:, 1] + r_target - 1.0)**2)
    
    # Overlap penalties
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    viol = np.maximum(0, 2.0 * r_target - dists)
    p += 0.5 * np.sum(viol**2)  # 0.5 to account for symmetric counting
    
    return mu * p

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    r_h = 0.1
    pts = []
    row = 0
    y = r_h
    while y + r_h <= 1.0 and len(pts) < n:
        shift = r_h if row % 2 else 0.0
        x = r_h + shift
        while x + r_h <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_h
        y += np.sqrt(3) * r_h
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    inits.append(np.array(pts[:n]))
    
    # 2. Grid layout
    g = np.linspace(0.12, 0.88, 5)
    pts2 = [[x, y] for y in g for x in g]
    pts2.append([0.5, 0.5])
    inits.append(np.array(pts2[:n]))
    
    # 3. Perturbed hex
    np.random.seed(42)
    for _ in range(6):
        inits.append(np.clip(inits[0] + np.random.uniform(-0.05, 0.05, (n, 2)), 0.05, 0.95))
        
    bounds_c = [(0.0, 1.0)] * (2 * n)
    
    # Optimize centers for each initialization
    for cfg in inits:
        x0 = cfg.flatten()
        
        # Stage 1: Gentle push to find basin
        res1 = minimize(penalty_obj, x0, args=(n, 50.0, 0.09), method='L-BFGS-B', bounds=bounds_c, options={'maxiter': 1500})
        # Stage 2: Strong push to target radius
        res2 = minimize(penalty_obj, res1.x, args=(n, 1500.0, 0.1015), method='L-BFGS-B', bounds=bounds_c, options={'maxiter': 2000})
        # Stage 3: Fine-tune with high penalty
        res3 = minimize(penalty_obj, res2.x, args=(n, 5000.0, 0.1025), method='L-BFGS-B', bounds=bounds_c, options={'maxiter': 3000})
        
        opt_c = res3.x.reshape(n, 2)
        radii, s = solve_radii_lp(opt_c)
        
        if radii is not None and s > best_sum:
            best_sum = s
            best_centers = opt_c
            best_radii = radii
            
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict numerical validity
    scale = 1.0
    for _ in range(100):
        ok = True
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                ok = False
                break
        if ok:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(best_centers[i] - best_centers[j])
                    if d < best_radii[i] + best_radii[j] - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            break
        scale *= 0.99995
        best_radii *= scale
        
    best_sum = float(np.sum(best_radii))
    return best_centers, best_radii, best_sum
