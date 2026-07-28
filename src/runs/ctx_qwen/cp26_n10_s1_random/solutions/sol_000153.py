# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000140 (state 07ed95ff) state=7776f0cc sum of radii=2.341611 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def obj_equal(vars):
    """Objective: maximize radius r => minimize -r"""
    return -vars[-1]

def con_equal(vars, n, triu_mask):
    """Inequality constraints for equal-radius packing >= 0"""
    r = vars[-1]
    c = vars[:2*n].reshape(n, 2)
    cx, cy = c[:, 0], c[:, 1]
    
    # Boundary constraints: x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0
    bc = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise separation: dist^2 >= 4r^2
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pc = dist_sq[triu_mask] - 4.0 * r**2
    
    return np.concatenate([bc, pc])

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    triu_mask = np.triu_indices(n, k=1)
    
    m = len(triu_mask[0])
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), triu_mask[0]] = 1.0
    A_ub[np.arange(m), triu_mask[1]] = 1.0
    b_ub = dists[triu_mask]
    
    c_obj = -np.ones(n)
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def generate_hex_init(rows, rng):
    """Generates a hexagonal lattice initialization for given row counts."""
    pts = []
    y = 0.12
    dy = 0.18
    dx = 0.21
    for i, cnt in enumerate(rows):
        shift = dx * 0.5 if i % 2 == 1 else 0.0
        x_start = (1.0 - (cnt - 1) * dx) / 2.0 + shift
        for _ in range(cnt):
            pts.append([x_start, y])
            x_start += dx
        y += dy
    res = np.array(pts[:26])
    res += rng.uniform(-0.01, 0.01, res.shape)
    return np.clip(res, 0.05, 0.95)

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    triu_mask = np.triu_indices(n, k=1)
    
    # Diverse row distributions that sum to 26
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [6, 6, 5, 5, 4],
        [4, 6, 6, 6, 4], [5, 5, 6, 5, 5], [5, 6, 5, 5, 5]
    ]
    
    best_equal_r = 0.0
    best_equal_centers = None
    configs = []
    
    # Generate structured starts
    for rd in row_dists:
        configs.append(generate_hex_init(rd, rng))
        
    # Add perturbed versions to escape symmetry traps
    for _ in range(8):
        base = configs[rng.integers(len(configs))]
        cfg = base + rng.uniform(-0.04, 0.04, (n, 2))
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # Phase 1: Optimize equal radius structure
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.09]])
        bounds_eq = [(0.0, 1.0)] * (2*n) + [(0.05, 0.12)]
        try:
            res = minimize(obj_equal, x0, args=(n, triu_mask), method='SLSQP',
                          bounds=bounds_eq, 
                          constraints={'type': 'ineq', 'fun': con_equal, 'args': (n, triu_mask)},
                          options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun) and res.x[-1] > best_equal_r:
                c_vals = con_equal(res.x, n, triu_mask)
                if np.min(c_vals) >= -1e-6:
                    best_equal_r = res.x[-1]
                    best_equal_centers = res.x[:2*n].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback grid if optimization fails
    if best_equal_centers is None:
        grid = np.array([[0.1 + i*0.18, 0.1 + j*0.18] for j in range(5) for i in range(5)])
        best_equal_centers = np.vstack([grid, [0.5, 0.5]])
        best_equal_r = 0.09
        
    # Phase 2: Unequal radius optimization via LP + Hill Climbing
    centers = best_equal_centers.copy()
    radii, current_sum = solve_radii_lp(centers)
    best_sum = current_sum
    best_final_centers = centers.copy()
    best_final_radii = radii.copy()
    
    step = 0.008
    for _ in range(1500):
        i = rng.integers(n)
        old_c = centers[i].copy()
        centers[i] += rng.uniform(-step, step, 2)
        centers[i] = np.clip(centers[i], 0.01, 0.99)
        
        new_radii, new_sum = solve_radii_lp(centers)
        if new_sum > best_sum + 1e-8:
            best_sum = new_sum
            best_final_centers = centers.copy()
            best_final_radii = new_radii.copy()
        else:
            centers[i] = old_c
        step *= 0.9995  # Gradually refine search
        
    # Phase 3: Strict numerical safety margin
    if best_final_radii is not None:
        best_final_radii *= 0.9999999
        best_sum = float(np.sum(best_final_radii))
        
    return best_final_centers, best_final_radii, best_sum
