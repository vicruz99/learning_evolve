# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=5009e3c8 sum of radii=2.240347 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def _get_pair_indices(n):
    """Precompute indices for all unique circle pairs."""
    return np.triu_indices(n, k=1)

def solve_radii_lp(centers, pair_i, pair_j):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Constraints: r_i + r_j <= dist_ij, r_i <= boundary margins, r_i >= 0
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    
    c_obj = np.ones(n) * -1.0
    
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs + 4 * n, n))
    b_ub = np.zeros(num_pairs + 4 * n)
    
    idx = 0
    # Pairwise distance constraints
    for i, j in zip(pair_i, pair_j):
        dist = np.hypot(x[i] - x[j], y[i] - y[j])
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dist
        idx += 1
        
    # Boundary constraints
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = x[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y[i]; idx += 1
        
    bounds = [(0.0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x
    return np.zeros(n)

def compute_constraints(v, n, pair_i, pair_j):
    """Vectorized inequality constraints for SLSQP."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    
    # Boundary constraints: >= 0
    cons = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2  =>  dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    pair_cons = (dist_sq[pair_i, pair_j] - r_sum[pair_i, pair_j]**2)
    
    return np.concatenate([cons, pair_cons])

def objective_func(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraint_fun(v, n, pair_i, pair_j):
    """Wrapper for constraint function to satisfy optimizer signature."""
    return compute_constraints(v, n, pair_i, pair_j)

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    pair_i, pair_j = _get_pair_indices(n)
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.25]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.25)] * n
    cons_dict = {'type': 'ineq', 'fun': lambda v: constraint_fun(v, n, pair_i, pair_j)}
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    r_h = 0.088
    y_val = r_h
    row = 0
    hex_pts = []
    while len(hex_pts) < n + 5:
        x_start = r_h if row % 2 == 0 else 2 * r_h
        x_val = x_start
        while x_val <= 1 - r_h:
            hex_pts.append([x_val, y_val])
            x_val += 2 * r_h
        y_val += r_h * np.sqrt(3)
        row += 1
    inits.append(np.array(hex_pts[:n]))
    
    # 2. 5x5 grid with jitter
    grid_pts = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)])
    while len(grid_pts) < n:
        grid_pts = np.vstack([grid_pts, [0.15, 0.15]])
    inits.append(grid_pts[:n])
    
    # 3. Boundary-hugging configuration
    bound_pts = np.zeros((n, 2))
    special = [[0.1,0.1],[0.9,0.1],[0.1,0.9],[0.9,0.9],
               [0.5,0.1],[0.5,0.9],[0.1,0.5],[0.9,0.5],
               [0.3,0.3],[0.7,0.3],[0.3,0.7],[0.7,0.7],
               [0.5,0.5],[0.2,0.5],[0.8,0.5],[0.5,0.2],[0.5,0.8],
               [0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75],
               [0.15,0.5],[0.85,0.5],[0.5,0.15],[0.5,0.85],
               [0.3,0.5],[0.7,0.5]]
    for idx, p in enumerate(special):
        if idx < n: bound_pts[idx] = p
    inits.append(bound_pts)
    
    # 4. Random seeds
    np.random.seed(42)
    for seed in [0, 13, 99]:
        np.random.seed(seed)
        inits.append(np.random.uniform(0.08, 0.92, size=(n, 2)))

    # Phase 1: Joint Optimization + LP Refinement
    for init_centers in inits:
        x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], np.full(n, 0.06)])
        
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP',
                           bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 1200, 'ftol': 1e-12, 'disp': False})
            cur_centers = res.x[:2*n].reshape(n, 2)
            cur_centers = np.clip(cur_centers, 0.0, 1.0)
        except Exception:
            cur_centers = init_centers.copy()
            
        # Phase 2: Exact LP for radii
        cur_radii = solve_radii_lp(cur_centers, pair_i, pair_j)
        cur_sum = np.sum(cur_radii)
        
        # Phase 3: Local search on centers to relax LP constraints
        improved = True
        search_iter = 0
        while improved and search_iter < 4:
            improved = False
            search_iter += 1
            for i in range(n):
                base_sum = cur_sum
                best_dx, best_dy = 0.0, 0.0
                
                # Candidate moves
                moves = [(0.015, 0), (-0.015, 0), (0, 0.015), (0, -0.015),
                         (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01), (-0.01, -0.01)]
                
                for dx, dy in moves:
                    temp_centers = cur_centers.copy()
                    temp_centers[i, 0] = np.clip(temp_centers[i, 0] + dx, 0.0, 1.0)
                    temp_centers[i, 1] = np.clip(temp_centers[i, 1] + dy, 0.0, 1.0)
                    temp_radii = solve_radii_lp(temp_centers, pair_i, pair_j)
                    temp_sum = np.sum(temp_radii)
                    
                    if temp_sum > base_sum + 1e-8:
                        base_sum = temp_sum
                        best_dx, best_dy = dx, dy
                        
                if best_dx != 0.0 or best_dy != 0.0:
                    cur_centers[i, 0] = np.clip(cur_centers[i, 0] + best_dx, 0.0, 1.0)
                    cur_centers[i, 1] = np.clip(cur_centers[i, 1] + best_dy, 0.0, 1.0)
                    cur_radii = solve_radii_lp(cur_centers, pair_i, pair_j)
                    cur_sum = np.sum(cur_radii)
                    improved = True
        
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = cur_centers.copy()
            best_radii = cur_radii.copy()
            
    # Final safety clamping to strictly satisfy validator tolerances
    centers = best_centers
    radii = best_radii
    
    for i in range(n):
        r = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = r
        
    # Resolve pairwise overlaps strictly (should be minimal due to LP)
    for _ in range(3):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d + 1e-9) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
