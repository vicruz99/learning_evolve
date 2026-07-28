# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=66083914 sum of radii=2.209131 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_eqr(vars_, n):
    """Minimize negative shared radius r."""
    return -vars_[2*n]

def constraints_eqr(vars_, n):
    """Inequality constraints >= 0 for valid equal-radius packing."""
    c = vars_[:2*n].reshape(n, 2)
    r = vars_[2*n]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    b = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    
    # Pairwise non-overlap: dist^2 >= 4r^2
    d2 = np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p = d2[mask] - 4.0 * r**2
    
    return np.concatenate([b, p])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_r = 0.0
    best_centers = None
    
    # Deterministic hexagonal row patterns summing to >= 26
    patterns = [
        [5, 5, 5, 5, 5, 1], [5, 5, 5, 5, 6], [5, 5, 5, 6, 5], [5, 5, 6, 5, 5], 
        [5, 6, 5, 5, 5], [6, 5, 5, 5, 5], [4, 6, 6, 6, 4], [5, 6, 6, 5, 4], 
        [6, 4, 6, 6, 4], [4, 5, 6, 6, 5], [5, 4, 6, 6, 5], [6, 5, 4, 6, 5]
    ]
    
    rng = np.random.default_rng(42)
    inits = []
    
    # Generate diverse initial configurations
    for pat in patterns:
        r0 = 0.095
        pts = []
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            if len(pts) >= n: break
        while len(pts) < n:
            pts.append([0.5, 0.5])
            
        cfg = np.array(pts[:n])
        # Add controlled perturbations to break symmetry
        for _ in range(4):
            p_cfg = cfg + rng.uniform(-0.015, 0.015, cfg.shape)
            p_cfg = np.clip(p_cfg, 0.02, 0.98)
            inits.append(p_cfg)
            
    bounds = [(0.0, 1.0)] * (2*n) + [(0.08, 0.12)]
    cons = {'type': 'ineq', 'fun': constraints_eqr, 'args': (n,)}
    
    # Stage 1: Optimize centers and shared radius
    for cfg in inits:
        x0 = np.concatenate([cfg.flatten(), [0.092]])
        try:
            res = minimize(objective_eqr, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            if res.success:
                r_val = res.x[2*n]
                if r_val > best_r:
                    best_r = r_val
                    best_centers = res.x[:2*n].reshape(n, 2).copy()
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = inits[0]
        
    # Stage 2: LP to maximize sum of individual radii for fixed centers
    c = best_centers
    walls = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                       np.minimum(c[:, 1], 1.0 - c[:, 1]))
    walls = np.maximum(walls, 1e-9) # Ensure positive bounds
    
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    # LP: maximize sum(r) <=> minimize -sum(r)
    c_obj = -np.ones(n)
    bounds_lp = [(0.0, w) for w in walls]
    
    m = pair_dists.shape[0]
    A_ub = np.zeros((m, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            idx += 1
            
    radii = None
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=pair_dists, bounds=bounds_lp, method='highs')
        if res_lp.success and np.all(res_lp.x >= 0):
            radii = res_lp.x
    except Exception:
        pass
        
    # If LP fails or yields invalid, fall back to safe equal radii
    if radii is None or np.any(radii < 0):
        min_wall = np.min(walls)
        min_pair = np.min(dists[mask]) / 2.0
        r_eq = min(min_wall, min_pair)
        radii = np.full(n, r_eq)
        
    # Final safety scaling to strictly satisfy the 1e-12 grader tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = c[i, 0], c[i, 1], radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            rs = radii[i] + radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    radii *= scale * 0.999998
    sum_radii = float(np.sum(radii))
    
    return c, radii, sum_radii
