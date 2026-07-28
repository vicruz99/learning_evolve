# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 81a0d5f4) state=d1d7ce9e sum of radii=1.447539 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty_and_obj(x, n, mu):
    """Computes negative sum of radii plus squared penalty for constraint violations."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    penalty = 0.0
    
    # Boundary penalties: circles must stay within [0, 1]
    penalty += np.sum(np.maximum(0, r - cx)**2)
    penalty += np.sum(np.maximum(0, r + cx - 1.0)**2)
    penalty += np.sum(np.maximum(0, r - cy)**2)
    penalty += np.sum(np.maximum(0, r + cy - 1.0)**2)
    
    # Overlap penalties: dist(i,j) >= r_i + r_j
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    dist_sq = cx_diff**2 + cy_diff**2
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    r_sum = r[:, None] + r[None, :]
    triu_idx = np.triu_indices(n, k=1)
    overlap = np.maximum(0, r_sum[triu_idx] - dist[triu_idx])
    penalty += np.sum(overlap**2)
    
    return obj + mu * penalty

def get_constraints_slscp(x, n):
    """Computes exact inequality constraints >= 0 for valid packing."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    cons = []
    # Boundary constraints
    cons.extend(cx - r)
    cons.extend(1.0 - cx - r)
    cons.extend(cy - r)
    cons.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    dist_sq = cx_diff**2 + cy_diff**2
    np.fill_diagonal(dist_sq, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    triu_idx = np.triu_indices(n, k=1)
    cons.extend(dist_sq[triu_idx] - (r_sum[triu_idx])**2)
    
    return np.array(cons)

def obj_sum_radii(x, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def get_hex_init(n, r_est):
    """Generates an initial hexagonal grid of n centers."""
    pts = []
    y = r_est
    row = 0
    dy = np.sqrt(3) * r_est
    while len(pts) < n:
        shift = r_est if row % 2 == 1 else 0.0
        x = r_est + shift
        while x + r_est <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_est
        y += dy
        row += 1
        if y > 1.0 - r_est:
            break
    # Fill remaining if lattice doesn't yield enough points
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Variable bounds: centers in [0, 1], radii in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattices with varying base densities
    for r_est in [0.08, 0.09, 0.10, 0.11]:
        configs.append(get_hex_init(n, r_est))
        
    # 2. Perturbed hexagonal configurations to escape symmetry
    for _ in range(5):
        c = get_hex_init(n, 0.09)
        c += np.random.uniform(-0.03, 0.03, c.shape)
        c = np.clip(c, 0.05, 0.95)
        configs.append(c)
        
    # 3. Random valid starts
    for _ in range(5):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # 4. Regular grid starts
    for s in [0.18, 0.20]:
        pts = []
        for i in range(6):
            for j in range(6):
                if len(pts) < n:
                    pts.append([s * (i + 0.5), s * (j + 0.5)])
        configs.append(np.array(pts[:n]))
        
    mu = 50000.0
    
    for cfg in configs:
        # Initialize radii small enough to be strictly valid for the current centers
        r0 = np.full(n, 0.08)
        for i in range(n):
            d_wall = min(cfg[i, 0], 1.0 - cfg[i, 0], cfg[i, 1], 1.0 - cfg[i, 1])
            d_pair = np.min(np.sqrt(np.sum((cfg - cfg[i])**2, axis=1)))
            r0[i] = min(0.12, d_wall / 2.0, d_pair / 2.0) * 0.85
            
        x0 = np.zeros(3 * n)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = r0
        
        try:
            # Phase 1: Robust exploration with penalty method
            res1 = minimize(compute_penalty_and_obj, x0, args=(n, mu),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 40000, 'ftol': 1e-15})
            
            cx_opt = res1.x[0::3]
            cy_opt = res1.x[1::3]
            r_opt = res1.x[2::3]
            
            # Scale to guarantee strict validity before polishing
            scale = 1.0
            for i in range(n):
                if r_opt[i] > 1e-9:
                    scale = min(scale, cx_opt[i] / r_opt[i], (1.0 - cx_opt[i]) / r_opt[i],
                               cy_opt[i] / r_opt[i], (1.0 - cy_opt[i]) / r_opt[i])
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt((cx_opt[i] - cx_opt[j])**2 + (cy_opt[i] - cy_opt[j])**2)
                    rs = r_opt[i] + r_opt[j]
                    if rs > 1e-9:
                        scale = min(scale, d / rs)
                        
            r_safe = r_opt * scale * 0.999999
            s_penalty = np.sum(r_safe)
            
            # Phase 2: Exact constraint refinement with SLSQP
            x0_slscp = np.concatenate([cx_opt, cy_opt, r_safe])
            try:
                res2 = minimize(obj_sum_radii, x0_slscp, args=(n,),
                               method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': get_constraints_slscp, 'args': (n,)},
                               options={'maxiter': 3000, 'ftol': 1e-14})
                
                cx_p = res2.x[0::3]
                cy_p = res2.x[1::3]
                r_p = res2.x[2::3]
                
                # Quick validation check for SLSQP result
                valid_slscp = True
                for i in range(n):
                    if cx_p[i] < r_p[i] or cx_p[i] > 1.0 - r_p[i] or cy_p[i] < r_p[i] or cy_p[i] > 1.0 - r_p[i]:
                        valid_slscp = False
                        break
                if valid_slscp:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = np.sqrt((cx_p[i] - cx_p[j])**2 + (cy_p[i] - cy_p[j])**2)
                            if d < r_p[i] + r_p[j] - 1e-9:
                                valid_slscp = False
                                break
                        if not valid_slscp:
                            break
                            
                if valid_slscp:
                    s_slscp = np.sum(r_p)
                    if s_slscp > best_sum:
                        best_sum = s_slscp
                        best_centers = np.column_stack((cx_p, cy_p))
                        best_radii = r_p
                    continue # Skip storing penalty result if polish succeeded and was better
            except Exception:
                pass
                
            # Fallback to penalty result if SLSQP fails or doesn't improve
            if s_penalty > best_sum:
                best_sum = s_penalty
                best_centers = np.column_stack((cx_opt, cy_opt))
                best_radii = r_safe
                
        except Exception:
            continue
            
    # Guaranteed fallback configuration
    if best_centers is None:
        r_fb = 0.095
        fb_centers = np.array([(i * 2 * r_fb + r_fb, j * 2 * r_fb + r_fb) 
                               for j in range(5) for i in range(5)] + [[0.55, 0.55]])
        fb_radii = np.full(26, r_fb)
        best_centers = fb_centers
        best_radii = fb_radii
        best_sum = np.sum(fb_radii)
        
    return best_centers, best_radii, float(best_sum)
