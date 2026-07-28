# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state bde5dee5) state=e6176fba sum of radii=1.890890 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n, mu, tri_i, tri_j):
    """
    Computes the objective: negative sum of radii + penalty for constraint violations.
    """
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    obj = -np.sum(r)
    penalty = 0.0
    
    # Boundary penalties: circles must stay within [0, 1]
    # Left: x - r >= 0
    v = c[:, 0] - r
    penalty += np.sum(np.maximum(0, -v)**2)
    # Right: 1 - x - r >= 0
    v = 1.0 - c[:, 0] - r
    penalty += np.sum(np.maximum(0, -v)**2)
    # Bottom: y - r >= 0
    v = c[:, 1] - r
    penalty += np.sum(np.maximum(0, -v)**2)
    # Top: 1 - y - r >= 0
    v = 1.0 - c[:, 1] - r
    penalty += np.sum(np.maximum(0, -v)**2)
    
    # Overlap penalties: dist(i,j) >= r_i + r_j
    # Vectorized distance calculation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    violations = r_sum[tri_i, tri_j] - dists[tri_i, tri_j]
    penalty += np.sum(np.maximum(0, violations)**2)
    
    return obj + mu * penalty

def run_packing() -> tuple:
    n = 26
    mu = 50000.0  # High penalty weight to enforce constraints strictly
    tri_i, tri_j = np.triu_indices(n, k=1)
    
    configs = []
    
    # 1. Hexagonal lattice initialization (optimal density structure)
    hex_c = []
    r0 = 0.10
    dy = np.sqrt(3) * r0
    dx = 2 * r0
    y = r0
    # Row counts that sum to 26 and fit well in a square
    counts = [5, 6, 5, 6, 4]
    for ri, cnt in enumerate(counts):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(hex_c) < n:
                hex_c.append([x, y])
            x += dx
        y += dy
    hex_c = np.array(hex_c[:n])
    # Normalize to fit comfortably inside [0, 1]
    hex_c = (hex_c - hex_c.min(axis=0)) / (hex_c.max(axis=0) - hex_c.min(axis=0)) * 0.8 + 0.1
    configs.append(hex_c)
    
    # 2. 5x5 Grid + 1 center circle (alternative dense structure)
    grid_c = []
    for i in range(5):
        for j in range(5):
            grid_c.append([0.1 + i*0.2, 0.1 + j*0.2])
    grid_c.append([0.5, 0.5])
    configs.append(np.array(grid_c[:n]))
    
    # 3. Perturbed versions to escape local minima
    np.random.seed(42)
    for _ in range(6):
        cfg = hex_c + np.random.normal(0, 0.025, size=hex_c.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    # Bounds: centers in [0, 1], radii in [small, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-5, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Optimize from each configuration
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), np.full(n, 0.09)])
        
        res = minimize(compute_objective, x0, args=(n, mu, tri_i, tri_j),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 15000, 'ftol': 1e-12})
                       
        # Accept if successful or if objective indicates large radii
        if res.success or res.fun < -2.5:
            c_opt = res.x[:2*n].reshape(n, 2)
            r_opt = res.x[2*n:]
            
            # Quick validity check to filter out failed optimizations
            valid = True
            for i in range(n):
                if c_opt[i,0] < r_opt[i] or c_opt[i,0] > 1-r_opt[i] or \
                   c_opt[i,1] < r_opt[i] or c_opt[i,1] > 1-r_opt[i]:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        if np.linalg.norm(c_opt[i]-c_opt[j]) < r_opt[i]+r_opt[j] - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                    
            if valid:
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
                    
    # Fallback if optimization somehow yields nothing valid (unlikely)
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(n, 0.08)
        
    # Safety scaling to guarantee strict validity for the validator
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r < 1e-9: continue
        scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            r_sum = best_radii[i] + best_radii[j]
            if r_sum > 1e-9:
                scale = min(scale, dist / r_sum)
                
    # Apply scaling with a tiny margin for numerical stability
    best_radii *= scale * 0.99999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
