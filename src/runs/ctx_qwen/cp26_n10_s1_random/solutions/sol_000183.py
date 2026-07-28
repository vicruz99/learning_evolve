# sol_000183 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=61ad20de sum of radii=2.626454 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Max radius limited by boundaries
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
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
            
    bounds = [(0.0, lim) for lim in lims]
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def objective(vars_array, n):
    """Objective: minimize negative sum of radii => maximize sum of radii."""
    return -np.sum(vars_array[2*n:])

def constraints(vars_array, n):
    """Returns inequality constraints >= 0 for valid packing."""
    c = vars_array[:2*n].reshape(n, 2)
    r = vars_array[2*n:]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    cons = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    
    # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu_indices(n, k=1)
    cons.append(d2[mask] - r_sum[mask]**2)
    
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # 1. Generate diverse initial configurations
    inits = []
    rng = np.random.default_rng(42)
    
    # Known effective row distributions for N=26 hex packing
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 6, 6, 4],
        [5, 5, 6, 5, 5], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [5, 4, 6, 5, 6], [6, 4, 5, 6, 5]
    ]
    
    for pat in patterns:
        pts = []
        y = 0.09
        for r_idx, cnt in enumerate(pat):
            shift = 0.09 if r_idx % 2 else 0.0
            x = 0.09 + shift
            for _ in range(cnt):
                if len(pts) < n:
                    pts.append([x, y])
                x += 0.18
            y += 0.09 * np.sqrt(3)
            
        cfg = np.array(pts[:n])
        # Center and scale to fit comfortably in [0,1]
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0))
        cfg = cfg * 0.88 + 0.06
        inits.append(cfg)
        
        # Add controlled perturbations to break symmetry
        for _ in range(4):
            p = cfg + rng.uniform(-0.025, 0.025, cfg.shape)
            inits.append(np.clip(p, 0.05, 0.95))
            
    # Random dense starts
    for _ in range(8):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # 2. Joint optimization phase
    for cfg in inits:
        # Get feasible radii first to guarantee SLSQP starts in feasible region
        r_init, _ = solve_lp_radii(cfg)
        x0 = np.concatenate([cfg.flatten(), r_init])
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP',
                          bounds=bounds_vars,
                          constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                          options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                # LP refinement guarantees optimal radii for these optimized centers
                r_lp, s_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass
            
    # 3. Iterative local search refinement
    if best_centers is not None:
        for iteration in range(20):
            improved = False
            # Try multiple perturbations
            for _ in range(6):
                pert = best_centers + rng.uniform(-0.006, 0.006, best_centers.shape)
                pert = np.clip(pert, 0.05, 0.95)
                r_lp, s_lp = solve_lp_radii(pert)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = pert.copy()
                    best_radii = r_lp.copy()
                    improved = True
                    
            if improved:
                # Polish the improved configuration with SLSQP
                x0 = np.concatenate([best_centers.flatten(), best_radii])
                try:
                    res = minimize(objective, x0, args=(n,), method='SLSQP',
                                  bounds=bounds_vars,
                                  constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                                  options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                    if np.isfinite(res.fun):
                        c_opt = res.x[:2*n].reshape(n, 2)
                        r_lp, s_lp = solve_lp_radii(c_opt)
                        if s_lp > best_sum:
                            best_sum = s_lp
                            best_centers = c_opt.copy()
                            best_radii = r_lp.copy()
                except Exception:
                    pass
                    
    # Fallback safety net
    if best_centers is None:
        cfg = np.array([[0.1 + 0.18*i, 0.1 + 0.18*j] for j in range(5) for i in range(5)])[::n*4+1] # dummy fallback
        best_centers = np.clip(get_hex_config(n, [5, 6, 5, 6, 4], 0.09), 0.1, 0.9)
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # 4. Final safety scaling to strictly satisfy 1e-12 validator tolerance
    scale = 1.0
    c = best_centers
    r = best_radii
    for i in range(n):
        if r[i] > 1e-12:
            scale = min(scale, c[i,0]/r[i], (1.0-c[i,0])/r[i], c[i,1]/r[i], (1.0-c[i,1])/r[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i]-c[j])
            r_sum = r[i] + r[j]
            if r_sum > 1e-12:
                scale = min(scale, d / r_sum)
                
    # Apply minimal margin
    r *= scale * 0.9999995
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum

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
