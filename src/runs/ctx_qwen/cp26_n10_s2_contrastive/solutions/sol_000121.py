# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=43901826 sum of radii=2.626891 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.001)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise non-overlap: hypot(dx, dy) >= r_i + r_j
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    return np.concatenate([c_dist, c_bound])

def generate_inits(rng):
    """Generate diverse structured and random initial configurations."""
    inits = []
    # Hexagonal patterns with varying spacing and offsets
    for sp in np.linspace(0.14, 0.20, 7):
        for off_x in [0.0, sp / 2.0]:
            for off_y in [0.0, sp * 0.866]:
                pts = []
                y = sp / 2.0 + off_y
                row = 0
                while len(pts) < N and y < 1.0 - sp / 2.0:
                    x = sp / 2.0 + off_x + (row % 2) * sp / 2.0
                    col = 0
                    while x < 1.0 - sp / 2.0 and len(pts) < N:
                        pts.append([x, y])
                        x += sp
                        col += 1
                    y += sp * 0.866
                    row += 1
                while len(pts) < N:
                    pts.append([0.5, 0.5])
                inits.append(np.array(pts[:N]))
                
    # Random uniform placements
    for _ in range(25):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng_main = np.random.default_rng(42)
    inits = generate_inits(rng_main)
    
    # Phase 1: Multi-start SLSQP with LP initialization
    for base in inits:
        c_init = base.copy()
        c_init = np.clip(c_init, 0.02, 0.98)
        r_init = solve_radii_lp(c_init) * 0.98  # Slight shrink ensures strict feasibility
        
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                curr_r = solve_radii_lp(curr_c)
                curr_s = np.sum(curr_r)
                
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            continue

    # Phase 2: Basin Hopping / Local Refinement to escape local minima
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(80):
            # Decaying noise schedule
            scale = 0.005 * np.exp(-step / 25.0) + 0.001
            c_pert = current_c + np.random.randn(N, 2) * scale
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = solve_radii_lp(c_pert)
            s_pert = np.sum(r_pert)
            
            # Greedy acceptance
            if s_pert > current_s:
                current_c, current_r, current_s = c_pert, r_pert, s_pert
                
                x0 = np.zeros(3*N)
                x0[0::3] = current_c[:, 0]
                x0[1::3] = current_c[:, 1]
                x0[2::3] = current_r * 0.98
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        c_ref = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_ref = solve_radii_lp(c_ref)
                        s_ref = np.sum(r_ref)
                        if s_ref > current_s:
                            current_c, current_r, current_s = c_ref, r_ref, s_ref
                            
                except Exception:
                    pass
                    
                if current_s > best_sum:
                    best_sum = current_s
                    best_centers = current_c.copy()
                    best_radii = current_r.copy()

    # Fallback safety net
    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, (N, 2))
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    final_sum = float(np.sum(r_final))
    return c_final, r_final, final_sum
