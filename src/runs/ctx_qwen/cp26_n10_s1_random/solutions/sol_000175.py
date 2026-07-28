# sol_000175 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=59ddcb1a sum of radii=2.323382 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_slsqp_constraints(vars_array, n):
    """Computes inequality constraints >= 0 for SLSQP."""
    cx = vars_array[:n]
    cy = vars_array[n:2 * n]
    r = vars_array[2 * n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    c = np.concatenate([c, dist[idx_i, idx_j] - r_sum[idx_i, idx_j]])
    
    return c

def objective_func(vars_array, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_array[2 * n:])

def solve_lp_for_radii(centers, n):
    """Given fixed centers, solve LP to maximize sum of radii."""
    c_obj = -np.ones(n)
    bounds = [(0.0, 0.5)] * n
    
    # Precompute distances and boundary limits
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    limits = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    limits = np.maximum(limits, 0.0)
    
    # Build A_ub and b_ub
    num_pairs = n * (n - 1) // 2
    num_cons = num_pairs + 4 * n
    A_ub = np.zeros((num_cons, n))
    b_ub = np.zeros(num_cons)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
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
            return res.x
    except Exception:
        pass
    return None

def center_penalty_objective(centers_flat, radii, n):
    """Penalty objective for optimizing centers given fixed radii."""
    centers = centers_flat.reshape(n, 2)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: penalty += (x - r)**2
        if 1.0 - x - r < 0: penalty += (1.0 - x - r)**2
        if y - r < 0: penalty += (y - r)**2
        if 1.0 - y - r < 0: penalty += (1.0 - y - r)**2
        
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            sum_r = radii[i] + radii[j]
            if d - sum_r < 0:
                penalty += (d - sum_r)**2
                
    return penalty

def generate_hex_init(n, r0, rng):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    while len(pts) < n and y + r0 < 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += dy
        row += 1
        
    # Fill remaining if any
    while len(pts) < n:
        pts.append(rng.uniform(0.2, 0.8, 2))
        
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    bounds_xy = [(0.0, 1.0)] * (2 * n)
    bounds_r = [(1e-6, 0.5)] * n
    full_bounds = bounds_xy + bounds_r
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # --- Phase 1: SLSQP from multiple diverse starts ---
    inits = []
    # Hexagonal patterns
    for r0 in [0.085, 0.090, 0.095, 0.100]:
        base = generate_hex_init(n, r0, rng)
        inits.append(base)
        for _ in range(4):
            pert = base + rng.uniform(-0.025, 0.025, base.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # Random starts
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    cons_dict = {'type': 'ineq', 'fun': get_slsqp_constraints, 'args': (n,)}
    
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.085
        
        try:
            res = minimize(
                objective_func, x0, args=(n,),
                method='SLSQP', bounds=full_bounds,
                constraints=cons_dict,
                options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_opt = res.x[2*n:]
                
                # Quick validation check
                valid = True
                if np.any(r_opt < 1e-9) or np.any(res.x[:2*n] < r_opt.reshape(-1, 1)) or \
                   np.any(1.0 - res.x[:2*n] < r_opt.reshape(-1, 1)):
                    valid = False
                    
                if valid:
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = generate_hex_init(n, 0.09, rng)
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # --- Phase 2: Alternating Refinement (Centers -> LP Radii) ---
    # This decouples the problem to escape local minima and squeeze more volume.
    for iteration in range(15):
        # 1. Optimize centers given current radii
        curr_r = best_radii.copy()
        curr_c_flat = best_centers.flatten()
        
        res_c = minimize(
            center_penalty_objective, curr_c_flat, args=(curr_r, n),
            method='L-BFGS-B', bounds=bounds_xy,
            options={'maxiter': 2000, 'ftol': 1e-14}
        )
        
        new_centers = res_c.x.reshape(n, 2)
        
        # 2. Solve LP for maximal radii given new centers
        lp_radii = solve_lp_for_radii(new_centers, n)
        
        if lp_radii is not None:
            new_sum = np.sum(lp_radii)
            if new_sum > best_sum:
                best_sum = new_sum
                best_centers = new_centers.copy()
                best_radii = lp_radii.copy()
                
        # Occasionally re-inject a tiny perturbation to avoid stagnation
        if iteration % 5 == 4:
            best_centers = np.clip(best_centers + rng.uniform(-0.002, 0.002, (n, 2)), 0.01, 0.99)
            
    # --- Phase 3: Final LP Projection & Safety Scaling ---
    # Ensure radii are exactly feasible for the final centers
    final_radii = solve_lp_for_radii(best_centers, n)
    if final_radii is None:
        final_radii = best_radii
        
    # Strict safety scaling to satisfy 1e-12 tolerance in validator
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], final_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = final_radii[i] + final_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    final_radii *= scale * 0.9999995
    best_sum = float(np.sum(final_radii))
    
    return best_centers, final_radii, best_sum
