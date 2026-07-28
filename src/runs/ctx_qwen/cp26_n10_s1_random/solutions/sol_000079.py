# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=dcf2b27a sum of radii=1.142631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Objective: minimize negative radius (equivalent to maximizing radius)"""
    return -v[-1]

def constraints(v):
    """
    Inequality constraints >= 0:
    - Boundary constraints: center +/- radius within [0, 1]
    - Non-overlap constraints: squared pairwise distance >= 4 * radius^2
    """
    n = N_CIRCLES
    xs = v[:n]
    ys = v[n:2*n]
    r = v[2*n]
    
    cons = []
    # Boundary constraints
    cons.append(xs - r)
    cons.append(1.0 - xs - r)
    cons.append(ys - r)
    cons.append(1.0 - ys - r)
    
    # Pairwise non-overlap constraints
    xs_col = xs[:, np.newaxis]
    ys_col = ys[:, np.newaxis]
    dist_sq = (xs_col - xs_col.T)**2 + (ys_col - ys_col.T)**2
    triu_idx = np.triu_indices(n, k=1)
    cons.append(dist_sq[triu_idx] - 4.0 * r**2)
    
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_r = 0.0
    best_centers = None
    
    np.random.seed(42)
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice initialization (optimal density structure)
    r_hex = 0.085
    pts = []
    y = r_hex
    row = 0
    while len(pts) < n:
        shift = r_hex if row % 2 == 1 else 0.0
        x = r_hex + shift
        while x + r_hex <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_hex
        y += r_hex * np.sqrt(3)
        row += 1
    configs.append(np.array(pts[:n]))
    
    # 2. Perturbed hexagonal configurations to escape local minima/symmetry
    for i in range(6):
        cfg = configs[0].copy()
        cfg += np.random.uniform(-0.02, 0.02, cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    # 3. Grid-like configuration
    grid = np.array([(0.1 + i*0.2, 0.1 + j*0.2) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
    grid += np.random.uniform(-0.01, 0.01, grid.shape)
    grid = np.clip(grid, 0.05, 0.95)
    configs.append(grid[:n])
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.11)]
    
    # Optimize from each configuration
    for cfg in configs:
        v0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints},
                          options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            
            r_opt = res.x[-1]
            if r_opt > best_r:
                best_r = r_opt
                best_centers = res.x[:2*n].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback in case all optimizations fail
    if best_centers is None:
        best_centers = configs[0]
        best_r = 0.085
        
    # Compute exact maximal valid radii for the optimized centers
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        min_d = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                    best_centers[i, 1], 1.0 - best_centers[i, 1])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0
        
    # Safety scaling to guarantee strict validity within checker tolerance
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = radii[i]
        if r > 1e-12:
            scale = min(scale, (x/r), (1-x)/r, (y/r), (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = radii[i] + radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    radii *= scale * 0.99999
    best_centers = np.clip(best_centers, 1e-9, 1 - 1e-9)
    
    return best_centers, radii, float(np.sum(radii))
