# sol_000143 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000106 (state f79bfb57) state=b6d82fdd sum of radii=2.605253 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def cons_equal(vars):
    """Constraints for equal-radius packing: boundaries and pairwise distances >= 2*t"""
    x = vars[:N]
    y = vars[N:2*N]
    t = vars[2*N]
    
    # Preallocate constraint array
    n_pairs = N * (N - 1) // 2
    c = np.empty(4 * N + n_pairs)
    
    # Boundary constraints: x >= t, 1-x >= t, y >= t, 1-y >= t
    c[:N] = x - t
    c[N:2*N] = 1.0 - x - t
    c[2*N:3*N] = y - t
    c[3*N:4*N] = 1.0 - y - t
    
    # Pairwise non-overlap: dist^2 >= 4*t^2
    cx = np.column_stack((x, y))
    dx = cx[:, 0, None] - cx[None, :, 0]
    dy = cx[:, 1, None] - cx[None, :, 1]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    
    mask = np.triu_indices(N, k=1)
    c[4*N:] = dist_sq[mask] - 4.0 * t**2
    return c

def obj_equal(vars):
    """Objective: maximize t => minimize -t"""
    return -vars[2*N]

def cons_var(vars):
    """Constraints for variable-radius packing"""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    n_pairs = N * (N - 1) // 2
    c = np.empty(4 * N + n_pairs)
    
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    cx = np.column_stack((x, y))
    dx = cx[:, 0, None] - cx[None, :, 0]
    dy = cx[:, 1, None] - cx[None, :, 1]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    mask = np.triu_indices(N, k=1)
    c[4*N:] = dist_sq[mask] - r_sum[mask]**2
    return c

def obj_var(vars):
    """Objective: maximize sum(r) => minimize -sum(r)"""
    return -np.sum(vars[2::3])

def run_packing():
    np.random.seed(42)
    
    # --- Generate diverse initial hexagonal configurations ---
    configs = []
    row_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5]
    ]
    
    for pat in row_patterns:
        if sum(pat) != 26: 
            continue
        r_init = 0.10
        pts = []
        y = r_init
        for ri, cnt in enumerate(pat):
            shift = r_init if ri % 2 == 1 else 0.0
            x = r_init + shift
            for _ in range(cnt):
                pts.append([x, y])
                x += 2.0 * r_init
            y += r_init * np.sqrt(3)
        
        cfg = np.array(pts[:N])
        # Normalize to [0.1, 0.9] to guarantee initial feasibility
        min_c = cfg.min(axis=0)
        max_c = cfg.max(axis=0)
        cfg = (cfg - min_c) / (max_c - min_c) * 0.8 + 0.1
        
        configs.append(cfg)
        # Add perturbations to escape symmetry traps
        for _ in range(6):
            pert = cfg + np.random.uniform(-0.02, 0.02, cfg.shape)
            configs.append(np.clip(pert, 0.05, 0.95))

    # --- Phase 1: Optimize Equal Radius ---
    best_t = 0.0
    best_eq_centers = None
    bounds_eq = [(0.0, 1.0)] * (2 * N) + [(0.05, 0.15)]
    
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res = minimize(obj_equal, x0, method='SLSQP', bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': cons_equal},
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success and res.x[2*N] > best_t:
                best_t = res.x[2*N]
                best_eq_centers = res.x[:2*N].reshape(N, 2)
        except Exception:
            continue
            
    if best_eq_centers is None:
        best_eq_centers = configs[0]
        best_t = 0.09

    # --- Phase 2: Optimize Variable Radii ---
    best_sum = -1.0
    best_centers = None
    best_radii = None
    bounds_var = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    
    # Start from best equal config and its perturbations
    phase2_starts = [best_eq_centers]
    for _ in range(8):
        pert = best_eq_centers + np.random.uniform(-0.008, 0.008, best_eq_centers.shape)
        phase2_starts.append(np.clip(pert, 0.02, 0.98))
        
    for cfg in phase2_starts:
        x0 = np.zeros(3 * N)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = best_t  # Initialize radii tightly at the optimized equal value
        
        try:
            res = minimize(obj_var, x0, method='SLSQP', bounds=bounds_var,
                           constraints={'type': 'ineq', 'fun': cons_var},
                           options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            
            # Verify constraint satisfaction within numerical tolerance
            c_vals = cons_var(res.x)
            if np.min(c_vals) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3])).copy()
                    best_radii = res.x[2::3].copy()
        except Exception:
            continue

    if best_centers is None:
        best_centers = best_eq_centers
        best_radii = np.full(N, best_t)
        best_sum = np.sum(best_radii)

    # --- Phase 3: Linear Programming Refinement for Radii ---
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_pairs = len(pairs)
    A_ub = np.zeros((num_pairs + 4 * N, N))
    b_ub = np.zeros(num_pairs + 4 * N)
    
    idx = 0
    for i, j in pairs:
        d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                     best_centers[i, 1] - best_centers[j, 1])
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(N):
        x, y = best_centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            best_radii = np.maximum(lp_res.x, 0.0) * 0.9999999
            best_sum = np.sum(best_radii)
    except Exception:
        pass
        
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
