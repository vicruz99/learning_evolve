# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=974c0793 sum of radii=2.489634 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_constraints(vars_flat, n):
    """
    Computes inequality constraints for equal-radius packing.
    Returns an array of constraint values (all must be >= 0).
    Constraints:
      - Boundary: x >= R, 1-x >= R, y >= R, 1-y >= R
      - Pairwise: ||c_i - c_j||^2 >= 4*R^2
    """
    xy = vars_flat[:2 * n].reshape(n, 2)
    R = vars_flat[2 * n]
    c = []
    
    # Boundary constraints
    c.append(xy[:, 0] - R)
    c.append(1.0 - xy[:, 0] - R)
    c.append(xy[:, 1] - R)
    c.append(1.0 - xy[:, 1] - R)
    
    # Pairwise squared distance constraints
    diff = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.append(dist_sq[mask] - 4.0 * R * R)
    
    return np.concatenate(c)

def obj_func(vars_flat):
    """Objective: maximize R => minimize -R"""
    return -vars_flat[2 * N_CIRCLES]

def get_init_configs():
    """Generates diverse initial configurations to escape local minima."""
    configs = []
    r0 = 0.101
    
    # 1. Hexagonal lattice pattern (5, 6, 5, 6, 4)
    pts = []
    rows = [5, 6, 5, 6, 4]
    y = r0
    row_idx = 0
    for cnt in rows:
        shift = r0 if row_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) < N_CIRCLES:
                pts.append([x, y])
            x += 2.0 * r0
        y += r0 * np.sqrt(3)
        row_idx += 1
        
    hex_pts = np.array(pts[:N_CIRCLES])
    # Ensure initial feasibility by clipping and setting safe radius later
    configs.append(np.clip(hex_pts, 0.06, 0.94))
    
    # 2. Perturbed versions
    np.random.seed(42)
    for _ in range(9):
        pert = configs[0].copy()
        pert += np.random.uniform(-0.03, 0.03, pert.shape)
        pert = np.clip(pert, 0.06, 0.94)
        configs.append(pert)
        
    # 3. Uniform grid fallback
    g = np.linspace(0.15, 0.85, 5)
    grid_pts = np.array([(x, y) for y in g for x in g])
    grid_pts = np.vstack([grid_pts, [[0.5, 0.5]]])
    configs.append(np.clip(grid_pts[:N_CIRCLES], 0.06, 0.94))
    
    return configs

def solve_lp_for_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = np.ones(n) * -1.0  # Minimize -sum(r) => Maximize sum(r)
    bounds = [(0.0, 0.5)] * n
    
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    m = len(pairs) + 4 * n
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i, j in pairs:
        d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(n):
        x, y = centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def run_packing():
    configs = get_init_configs()
    best_R = 0.0
    best_centers = None
    
    bounds_vars = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.01, 0.15)]
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (N_CIRCLES,)}
    
    # Phase 1: Optimize centers and shared radius R
    for cfg in configs:
        x0 = np.zeros(2 * N_CIRCLES + 1)
        x0[:2 * N_CIRCLES] = cfg.flatten()
        x0[2 * N_CIRCLES] = 0.06  # Start with feasible small radius
        
        try:
            res = minimize(obj_func, x0, method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            
            # Check if optimization improved R and constraints are satisfied
            if -res.fun > best_R:
                cvals = compute_constraints(res.x, N_CIRCLES)
                if np.min(cvals) > -1e-6:
                    best_R = -res.fun
                    best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2).copy()
        except Exception:
            continue
            
    # Fallback if optimizer failed
    if best_centers is None:
        best_centers = configs[0]
        best_R = 0.06
        
    # Phase 2: Linear Programming to maximize sum of individual radii
    radii, sum_r = solve_lp_for_radii(best_centers)
    
    if radii is not None:
        # Apply tiny safety margin to strictly satisfy 1e-12 validation tolerance
        radii *= 0.99999999
        sum_r = float(np.sum(radii))
    else:
        # Fallback to equal radii from Phase 1
        radii = np.full(N_CIRCLES, best_R * 0.99999999)
        sum_r = float(np.sum(radii))
        
    return best_centers, radii, sum_r
