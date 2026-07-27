# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 634a18b7) state=545e7919 sum of radii=2.436330 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_hex_init(n=26):
    """Generate initial positions on a hexagonal lattice."""
    pts = []
    r = 0.09
    dy = np.sqrt(3) * r
    y = r
    toggle = False
    while len(pts) < n:
        x = r if not toggle else r + r
        while x + r <= 1.0:
            pts.append([x, y])
            x += 2 * r
            if len(pts) >= n:
                break
        y += dy
        toggle = not toggle
        if y + r > 1.0:
            # Fallback filler if hex grid runs out of vertical space
            while len(pts) < n:
                pts.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
            break
    return np.array(pts[:n])

def compute_objective(vars, n, lam):
    """Objective function: minimize negative sum of radii + penalty for violations."""
    c = vars.reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    val = -np.sum(r)
    pen = 0.0
    
    # Pairwise overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - r[i] - r[j]
            if gap < 0:
                pen += gap ** 2
                
    # Boundary penalty
    for i in range(n):
        pen += max(0, r[i] - x[i]) ** 2
        pen += max(0, r[i] - y[i]) ** 2
        pen += max(0, x[i] + r[i] - 1.0) ** 2
        pen += max(0, y[i] + r[i] - 1.0) ** 2
        
    # Negative radius penalty
    pen += np.sum(np.maximum(0, -r) ** 2) * 100.0
        
    return val + lam * pen

def run_packing():
    n = 26
    best_sum = -np.inf
    best_state = None
    
    # Multiple restarts to escape local optima
    for seed in range(8):
        np.random.seed(seed * 17 + 7)
        init_centers = get_hex_init(n)
        # Add controlled jitter to break symmetries
        init_centers += np.random.uniform(-0.015, 0.015, init_centers.shape)
        init_centers = np.clip(init_centers, 0.05, 0.95)
        
        init_radii = np.full(n, 0.075)
        x0 = np.column_stack([init_centers, init_radii]).flatten()
        
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.25)])
            
        try:
            res = minimize(compute_objective, x0, args=(n, 3000.0), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            cur_r = res.x.reshape(-1, 3)[:, 2]
            if np.sum(cur_r) > best_sum:
                best_sum = np.sum(cur_r)
                best_state = res.x.copy()
        except Exception:
            continue
            
    if best_state is None:
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    centers = best_state.reshape(-1, 3)[:, :2].copy()
    radii = best_state.reshape(-1, 3)[:, 2].copy()
    
    # Strict boundary enforcement
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        
    # Overlap resolution: iteratively shrink radii until valid
    for _ in range(1000):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if d < radii[i] + radii[j] - 1e-10:
                    diff = radii[i] + radii[j] - d
                    radii[i] -= diff / 2.0
                    radii[j] -= diff / 2.0
                    changed = True
        if not changed:
            break
            
    final_sum = np.sum(radii)
    return centers, radii, final_sum
