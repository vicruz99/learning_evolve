# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000082 (state 4b2dee7c) state=78e588e5 sum of radii=2.630172 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)
NUM_PAIRS = len(I)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I] = 1.0
    A_ub[np.arange(NUM_PAIRS), J] = 1.0
    
    dx = centers[I, 0] - centers[J, 0]
    dy = centers[I, 1] - centers[J, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds_r = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds_r.append((0.0, max(0.0, mx)))
        
    # Try HiGHS first (fastest), fallback to interior-point
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method=method)
            if res.success:
                return np.maximum(res.x, 0.0), -res.fun
        except Exception:
            continue
    return np.zeros(n), 0.0

def objective_slsqp(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_slsqp(x):
    """
    Inequality constraints (must be >= 0):
    - Pairwise distance >= sum of radii
    - Boundary clearance: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    # Use hypot for stable, non-vanishing gradients at contact points
    c_dist = np.hypot(dx, dy) - (r[I] + r[J])
    
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    return np.concatenate([c_dist, c_bound])

def generate_initializations():
    """Generate diverse initial center configurations."""
    inits = []
    rng_global = np.random.RandomState(42)
    
    # 1. Hexagonal lattices with varying spacing
    for sp in np.linspace(0.16, 0.21, 12):
        c = np.zeros((N, 2))
        idx = 0
        row = 0
        y = sp / 2
        while idx < N and y < 1.0 - sp / 2:
            x_start = sp / 2 + (row % 2) * sp / 2
            col = 0
            while x_start + col * sp < 1.0 - sp / 2 and idx < N:
                c[idx] = [x_start + col * sp, y]
                idx += 1
                col += 1
            y += sp * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = rng_global.uniform(0.2, 0.8, 2)
            idx += 1
        c += rng_global.normal(0, 0.005, c.shape)
        inits.append(np.clip(c, 0.05, 0.95))
        
    # 2. Specific row patterns optimized for N=26
    patterns = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 6, 5, 5], [5, 5, 5, 5, 6], [6, 6, 6, 4, 4]]
    for pat in patterns:
        c = np.zeros((N, 2))
        idx = 0
        y = 0.08
        dy = 0.18
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.045
            x = 0.05 + shift
            for _ in range(cnt):
                if idx < N:
                    c[idx] = [x, y]
                    idx += 1
                x += 0.165
            y += dy
        while idx < N:
            c[idx] = rng_global.uniform(0.3, 0.7, 2)
            idx += 1
        c += rng_global.normal(0, 0.008, c.shape)
        inits.append(np.clip(c, 0.05, 0.95))
        
    # 3. Random uniform placements
    for _ in range(15):
        c = rng_global.uniform(0.15, 0.85, (N, 2))
        inits.append(c)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints_slsqp}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multi-start SLSQP
    inits = generate_initializations()
    for c0 in inits:
        r0, _ = solve_lp_radii(c0)
        # Scale down slightly to guarantee strict feasibility for SLSQP start
        r0 = np.maximum(r0 * 0.92, 1e-4)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            curr_centers = np.column_stack((cx, cy))
            
            # Exact LP refinement extracts true maximal radii for optimized centers
            r_opt, s_opt = solve_lp_radii(curr_centers)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = curr_centers.copy()
                best_radii = r_opt.copy()
        except Exception:
            continue

    # Phase 2: Adaptive Basin Hopping with LP evaluation & SLSQP polishing
    if best_centers is not None:
        rng = np.random.RandomState(99)
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(180):
            # Decay noise scale, but keep a floor for exploration
            noise_scale = 0.006 * np.exp(-step / 60.0) + 0.0008
            c_pert = current_c + rng.normal(0, noise_scale, (N, 2))
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            if s_pert > current_s:
                # Successful jump: polish with SLSQP
                x0 = np.zeros(3 * N)
                x0[0::3] = c_pert[:, 0]
                x0[1::3] = c_pert[:, 1]
                x0[2::3] = np.maximum(r_pert * 0.95, 1e-4)
                
                try:
                    res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds,
                                   constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    c_new = np.column_stack((cx, cy))
                    r_new, s_new = solve_lp_radii(c_new)
                    
                    if s_new > current_s:
                        current_c, current_r, current_s = c_new, r_new, s_new
                        best_centers, best_radii, best_sum = current_c.copy(), current_r.copy(), current_s
                except Exception:
                    pass
            # Simulated annealing style acceptance occasionally to escape shallow basins
            elif s_pert > current_s * 0.998 and rng.rand() < 0.05:
                current_c, current_r, current_s = c_pert, r_pert, s_pert

    # Fallback safety net
    if best_centers is None:
        best_centers = np.random.uniform(0.2, 0.8, (N, 2))
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Strict post-processing to guarantee validity
    radii = best_radii.copy()
    centers = best_centers.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(radii[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.hypot(dx, dy)
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc / 2.0
                    radii[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    return centers, radii, final_sum
