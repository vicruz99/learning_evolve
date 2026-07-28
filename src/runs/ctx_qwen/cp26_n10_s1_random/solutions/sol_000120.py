# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000066 (state 7dd8b726) state=d6ae462d sum of radii=1.468945 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """
    Solves the Linear Program to maximize sum of radii for fixed centers.
    Constraints: r_i >= 0, r_i <= dist_to_wall, r_i + r_j <= dist(i, j)
    """
    n = centers.shape[0]
    # Distance to nearest boundary
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    wall = np.maximum(wall, 0.0)
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx = np.triu_indices(n, k=1)
    d_pairs = dists[idx]
    
    # LP: minimize -sum(r) => maximize sum(r)
    c_obj = -np.ones(n)
    bounds = [(0.0, w) for w in wall]
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n + n_pairs, n))
    b_ub = np.zeros(n + n_pairs)
    
    # Wall constraints: r_i <= wall_i
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = wall[i]
        
    # Pairwise constraints: r_i + r_j <= d_ij
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        A_ub[n + k, i] = 1.0
        A_ub[n + k, j] = 1.0
        b_ub[n + k] = d_pairs[k]
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x
    except Exception:
        pass
    return None

def constraint_func(vars_array, n, triu_idx):
    """
    Computes inequality constraints >= 0 for valid packing.
    Returns concatenated array of boundary and pairwise non-overlap constraints.
    """
    cx = vars_array[0::3]
    cy = vars_array[1::3]
    r = vars_array[2::3]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Pairwise squared distance constraints: dist^2 >= (ri + rj)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = r[:, None] + r[None, :]
    
    c = np.concatenate([c, (dx[triu_idx])**2 + (dy[triu_idx])**2 - dr[triu_idx]**2])
    return c

def objective(vars_array):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_array[2::3])

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    
    # Variable bounds: x,y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n, triu_idx)}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # --- Generate Initial Configurations ---
    configs = []
    
    # 1. Hexagonal lattice
    pts = []
    y = 0.1
    row = 0
    while len(pts) < n + 5:
        shift = 0.1 if row % 2 == 1 else 0.0
        x = 0.1 + shift
        while x + 0.1 <= 1.0:
            pts.append([x, y])
            x += 0.2
        y += 0.1 * np.sqrt(3)
        row += 1
    configs.append(np.array(pts[:n]))
    
    # 2. Perturbed hex lattices
    for _ in range(5):
        pert = configs[0] + rng.uniform(-0.04, 0.04, (n, 2))
        configs.append(np.clip(pert, 0.05, 0.95))
        
    # 3. Uniform grid + center
    gx = np.linspace(0.15, 0.85, 5)
    gy = np.linspace(0.15, 0.85, 5)
    grid_pts = np.array([(x, y) for y in gy for x in gx])
    grid_pts = np.vstack([grid_pts, [[0.5, 0.5]]])
    configs.append(grid_pts)
    
    # 4. Random valid starts
    for _ in range(4):
        rc = rng.uniform(0.1, 0.9, (n, 2))
        configs.append(rc)
        
    # --- Optimization Loop ---
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = 0.09 + rng.uniform(0.0, 0.02, n)
        
        try:
            # Phase 1: Joint optimization of centers and radii
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                
                # Phase 2: LP refinement to extract maximal radii for these centers
                r_opt = solve_lp_radii(c_opt)
                if r_opt is not None:
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
                        
                        # Phase 3: Refine centers using LP radii as initial guess
                        x0_refine = np.zeros(3 * n)
                        x0_refine[0::3] = c_opt[:, 0]
                        x0_refine[1::3] = c_opt[:, 1]
                        x0_refine[2::3] = r_opt
                        
                        res2 = minimize(objective, x0_refine, method='SLSQP', bounds=bounds,
                                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
                        
                        if np.isfinite(res2.fun):
                            c_ref = res2.x[:2*n].reshape(n, 2)
                            r_ref = solve_lp_radii(c_ref)
                            if r_ref is not None:
                                s_ref = np.sum(r_ref)
                                if s_ref > best_sum:
                                    best_sum = s_ref
                                    best_centers = c_ref.copy()
                                    best_radii = r_ref.copy()
        except Exception:
            continue
            
    # Fallback if optimization yields no valid result
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to strictly satisfy the validator's 1e-12 tolerance
    best_radii *= 0.999999
    best_sum = float(np.sum(best_radii))
    
    # Ensure centers are strictly within bounds after scaling
    best_centers = np.clip(best_centers, best_radii[:, np.newaxis], 1.0 - best_radii[:, np.newaxis])
    
    return best_centers, best_radii, best_sum
