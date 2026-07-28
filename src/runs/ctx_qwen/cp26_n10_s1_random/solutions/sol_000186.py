# sol_000186 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=ffa01a54 sum of radii=2.624552 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_hex_config(n, row_counts, r_init):
    """Generates a hexagonal lattice initialization with specified row counts."""
    pts = []
    y = r_init
    row_idx = 0
    for cnt in row_counts:
        shift = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(cnt):
            if len(pts) < n:
                pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
    return np.array(pts[:n])

def objective_joint(x, n):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def constraints_joint(x, n):
    """Returns inequality constraints >= 0 for valid packing."""
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    cons = []
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(d2[mask] - r_sum_sq[mask])
    return np.concatenate(cons)

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Boundary limits for each circle
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    
    c_obj = np.ones(n) * -1.0
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)  # Avoid self-constraints
    
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
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    # Fallback to tiny positive radii if LP fails
    return np.full(n, 1e-6), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints_joint, 'args': (n,)}
    
    # Diverse row distributions for hexagonal packing
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 6, 6, 4],
        [5, 5, 6, 5, 5], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 4, 6, 6, 5], [6, 4, 6, 5, 5], [5, 6, 4, 5, 6],
        [4, 5, 6, 5, 6], [6, 5, 4, 5, 6]
    ]
    
    configs = []
    for rd in row_dists:
        if sum(rd) < n: 
            continue
        cfg = get_hex_config(n, rd, 0.1)
        configs.append(cfg)
        
    # Rotated hex grids to break symmetry and explore different alignments
    for angle in [0.05, -0.05, 0.1, -0.1, 0.15]:
        base = get_hex_config(n, [5, 6, 5, 6, 4], 0.1)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        cfg = base @ rot
        # Normalize and scale to fit comfortably inside [0,1]
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0))
        cfg = cfg * 0.9 + 0.05
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # Random dense starts
    rng = np.random.default_rng(42)
    for _ in range(8):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Joint SLSQP optimization from multiple starts
    for cfg in configs:
        r0 = np.full(n, 0.09)
        x0 = np.concatenate([cfg.flatten(), r0])
        try:
            res = minimize(objective_joint, x0, method='SLSQP', args=(n,),
                          bounds=bounds_vars, constraints=cons_dict,
                          options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                # LP refinement guarantees optimal radii for these centers
                r_lp, s_lp = solve_radii_lp(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_centers is not None:
        for _ in range(25):
            pert = best_centers + rng.uniform(-0.003, 0.003, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r0 = np.full(n, 0.09)
            x0 = np.concatenate([pert.flatten(), r0])
            try:
                res = minimize(objective_joint, x0, method='SLSQP', args=(n,),
                              bounds=bounds_vars, constraints=cons_dict,
                              options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if np.isfinite(res.fun):
                    c_opt = res.x[:2*n].reshape(n, 2)
                    r_lp, s_lp = solve_radii_lp(c_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass
                
    # Fallback safety net
    if best_centers is None:
        best_centers = np.clip(get_hex_config(n, [5, 6, 5, 6, 4], 0.09), 0.1, 0.9)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Phase 3: Final safety scaling to strictly satisfy 1e-12 validator tolerance
    scale = 1.0
    c = best_centers
    r = best_radii
    
    for i in range(n):
        if r[i] > 1e-12:
            scale = min(scale, c[i,0]/r[i], (1-c[i,0])/r[i], c[i,1]/r[i], (1-c[i,1])/r[i])
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i]-c[j])
            r_sum = r[i] + r[j]
            if r_sum > 1e-12:
                scale = min(scale, d / r_sum)
                
    # Apply with minimal margin to guarantee strict validity
    r *= scale * 0.9999995
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum
