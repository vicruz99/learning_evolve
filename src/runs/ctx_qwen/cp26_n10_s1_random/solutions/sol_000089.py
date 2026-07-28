# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=6d1b3f3d sum of radii=1.039990 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def objective(vars):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(vars[2::3])

def compute_constraints(vars):
    """
    Computes inequality constraints >= 0 for valid packing.
    Variables layout: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    """
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    c_list = []
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    c_list.append(x - r)          # x - r >= 0
    c_list.append(1.0 - x - r)    # 1 - x - r >= 0
    c_list.append(y - r)          # y - r >= 0
    c_list.append(1.0 - y - r)    # 1 - y - r >= 0
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dists = np.sqrt(dx**2 + dy**2)
    r_sums = r[i_idx] + r[j_idx]
    c_list.append(dists - r_sums)
    
    return np.concatenate(c_list)

def generate_hex_config(row_counts, r_init):
    """Generates initial positions on a hexagonal lattice with specified row distribution"""
    n = 26
    pts = []
    y = r_init
    row_idx = 0
    for count in row_counts:
        shift = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(count):
            if len(pts) < n:
                pts.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    # Generate diverse initial configurations based on hexagonal lattices
    init_configs = []
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4], 
        [5, 6, 6, 5, 4], [6, 5, 5, 6, 4], [5, 6, 5, 5, 5]
    ]
    
    for rd in row_dists:
        cfg = generate_hex_config(rd, 0.09)
        # Normalize and scale to center comfortably inside [0,1]
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0)) * 0.8 + 0.1
        init_configs.append(cfg)
        
    # Add perturbed versions to break symmetry
    for _ in range(5):
        cfg = init_configs[0].copy()
        cfg += np.random.uniform(-0.02, 0.02, cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        init_configs.append(cfg)
        
    # Optimize each configuration with iterative refinement
    for cfg in init_configs:
        current_c = cfg.copy()
        current_r = np.full(n, 0.08)
        
        for step in range(4):
            x0 = np.concatenate([current_c.flatten(), current_r])
            
            try:
                res = opt.minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': compute_constraints},
                    options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False}
                )
                
                if res.success and np.isfinite(res.fun):
                    c_opt = res.x[:2*n].reshape(n, 2)
                    r_opt = res.x[2*n:]
                    
                    # Quick validity check before accepting
                    valid = True
                    if np.any(c_opt[:, 0] < r_opt) or np.any(c_opt[:, 0] > 1 - r_opt): valid = False
                    if np.any(c_opt[:, 1] < r_opt) or np.any(c_opt[:, 1] > 1 - r_opt): valid = False
                    
                    if valid:
                        i_idx, j_idx = np.triu_indices(n, k=1)
                        dx = c_opt[i_idx, 0] - c_opt[j_idx, 0]
                        dy = c_opt[i_idx, 1] - c_opt[j_idx, 1]
                        dists = np.sqrt(dx**2 + dy**2)
                        r_sums = r_opt[i_idx] + r_opt[j_idx]
                        if np.any(dists < r_sums - 1e-9):
                            valid = False
                            
                    if valid:
                        s = np.sum(r_opt)
                        if s > best_sum:
                            best_sum = s
                            best_centers = c_opt.copy()
                            best_radii = r_opt.copy()
                            
                        # Perturb for next iterative refinement step
                        current_c = c_opt + np.random.uniform(-0.004, 0.004, (n, 2))
                        current_c = np.clip(current_c, 0.01, 0.99)
                        current_r = r_opt * (1 + np.random.uniform(-0.015, 0.015, n))
                        current_r = np.clip(current_r, 1e-6, 0.49)
                    else:
                        break
                else:
                    break
            except Exception:
                break
                
    # Fallback configuration if optimization yields nothing
    if best_centers is None:
        best_centers = generate_hex_config([5, 6, 5, 6, 4], 0.08)
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        scale = min(scale, 
                    best_centers[i, 0]/best_radii[i], 
                    (1-best_centers[i, 0])/best_radii[i],
                    best_centers[i, 1]/best_radii[i], 
                    (1-best_centers[i, 1])/best_radii[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d > 0:
                scale = min(scale, d / (best_radii[i] + best_radii[j]))
                
    best_radii *= scale * 0.99999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
