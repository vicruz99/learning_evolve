# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=38f871d6 sum of radii=2.623431 correctness=1.0
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
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    # Boundary constraints handled via bounds for efficiency
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattices with varying spacing
    for s in np.linspace(0.14, 0.21, 10):
        c = np.zeros((N, 2))
        idx = 0
        y = s / 2
        row = 0
        while idx < N and y < 1.0 - s / 2:
            x = s / 2 + (row % 2) * s / 2
            while x < 1.0 - s / 2 and idx < N:
                c[idx] = [x, y]
                x += s
                idx += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.2, 0.8, 2)
            idx += 1
        c += rng.normal(0, 0.005, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # Staggered row patterns known to yield high-density packings
    patterns = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4], [4, 5, 6, 5, 6]]
    for pat in patterns:
        for spacing in np.linspace(0.14, 0.19, 4):
            pts = []
            y = spacing * 0.6
            dy = spacing * np.sqrt(3) / 2 * 0.98
            for r_idx, cnt in enumerate(pat):
                shift = 0.0 if r_idx % 2 == 0 else spacing / 2.0
                x = spacing * 0.6 + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += spacing
                y += dy
            while len(pts) < N:
                pts.append([0.5, 0.5])
            inits.append(np.array(pts[:N]) + rng.normal(0, 0.005, (N, 2)))
            
    # Random uniform placements
    for _ in range(25):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # Phase 1: Multi-start SLSQP optimization
    for c_init in inits:
        r_init, _ = solve_radii_lp(c_init)
        r_init = np.maximum(r_init * 0.95, 1e-5)
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                c_opt = np.column_stack((cx, cy))
                r_opt, s_opt = solve_radii_lp(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Adaptive Basin Hopping on centers with LP radii extraction
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        stagnation = 0
        
        for step in range(600):
            # Adaptive noise schedule that decays over time
            noise_scale = 0.004 * max(0.2, 1.0 - step / 500.0)
            c_pert = curr_c + rng.normal(0, noise_scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_radii_lp(c_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                stagnation = 0
                
                # Local SLSQP polish after successful jump
                x0 = np.zeros(3*N)
                x0[0::3] = curr_c[:, 0]
                x0[1::3] = curr_c[:, 1]
                x0[2::3] = np.maximum(curr_r * 0.98, 1e-5)
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_pol, s_pol = solve_radii_lp(c_pol)
                        if s_pol > curr_s:
                            curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                except Exception:
                    pass
                    
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
            else:
                stagnation += 1
                # Occasional full restart if heavily stuck to escape deep local minima
                if stagnation > 150 and rng.random() < 0.1:
                    curr_c = rng.uniform(0.1, 0.9, (N, 2))
                    curr_r, curr_s = solve_radii_lp(curr_c)

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_radii_lp(best_centers)
        
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
    for _ in range(60):
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
