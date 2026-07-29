# sol_000083 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000039 (state 91d6f1d3) state=035f7524 sum of radii=2.588135 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints: boundary clearance and pairwise non-overlap.
    Returns array of constraint values (must be >= 0).
    """
    cx, cy, r = x[0::3], x[1::3], x[2::3]
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    dists = np.hypot(dx, dy)
    c_overlap = dists - (r[I_IDX] + r[J_IDX])
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_overlap, c_bound])

def solve_lp_radii(centers):
    """
    Given fixed centers, solve an LP to find radii that maximize the sum of radii
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Precompute pairwise distances for LP bounds
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    # Construct sparse-like A_ub for r_i + r_j <= d_ij
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    # Variable bounds: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    # Try HiGHS first, fallback to interior-point
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return res.x, -res.fun
        except Exception:
            continue
            
    return np.zeros(n), 0.0

def generate_initial_configs(num_configs):
    """Create diverse starting center configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying spacing and offsets
    for i in range(num_configs // 2):
        scale = 0.15 + 0.045 * (i / max(1, (num_configs // 2) - 1))
        pts = []
        y = scale + 0.05
        col = 0
        while len(pts) < N:
            x = scale + 0.05 + (col % 2) * scale
            while x <= 1.0 - scale - 0.05 and len(pts) < N:
                pts.append([x, y])
                x += 2 * scale
            y += scale * np.sqrt(3) / 2.0
            col += 1
        pts = np.array(pts[:N])
        # Add controlled noise to break symmetry
        rng = np.random.RandomState(i + 42)
        pts += rng.normal(0, 0.008, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        configs.append(pts)
        
    # 2. Random uniform placements
    for i in range(num_configs // 2, num_configs):
        rng = np.random.RandomState(i + 100)
        pts = rng.uniform(0.1, 0.9, (N, 2))
        configs.append(pts)
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Broad search from diverse initializations
    inits = generate_initial_configs(40)
    
    for c0 in inits:
        r0 = np.full(N, 0.03)  # Feasible small start radii
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            
            if res.success:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                # Refine radii exactly using LP
                r_lp, s_lp = solve_lp_radii(curr_c)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = curr_c.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement around the best solution
    if best_centers is not None:
        for trial in range(50):
            rng = np.random.RandomState(trial * 31 + 7)
            c_pert = best_centers + rng.normal(0, 0.0035, best_centers.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            # Get valid radii for perturbed centers
            r_pert, _ = solve_lp_radii(c_pert)
            r_pert = np.maximum(r_pert, 1e-5)
            
            x_pert = np.zeros(3 * N)
            x_pert[0::3] = c_pert[:, 0]
            x_pert[1::3] = c_pert[:, 1]
            x_pert[2::3] = r_pert
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds_opt,
                               constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_lp, s_lp = solve_lp_radii(curr_c)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = curr_c.copy()
                        best_radii = r_lp.copy()
            except Exception:
                continue

    # Fallback safety net
    if best_centers is None:
        rng = np.random.RandomState(0)
        best_centers = rng.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Strict post-processing to guarantee validity
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        if r_final[i] > mx:
            r_final[i] = max(0.0, mx - 1e-9)
            
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], 
                             best_centers[i,1]-best_centers[j,1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    overlap = r_final[i] + r_final[j] - d
                    r_final[i] -= overlap / 2.0
                    r_final[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return best_centers, r_final, float(np.sum(r_final))
