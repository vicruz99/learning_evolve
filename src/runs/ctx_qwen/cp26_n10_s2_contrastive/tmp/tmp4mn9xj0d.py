import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant structure for LP pairwise constraints
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 1e-6)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def generate_inits(rng):
    """Generate diverse structured and random initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying spacing and offsets
    for seed in range(20):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        s = 0.13 + r_gen.uniform(0, 0.08)
        y = s/2
        row = 0
        while idx < N and y < 1.0 - s/2:
            x_start = s/2 + (row % 2) * s/2
            col = 0
            while x_start + col*s < 1.0 - s/2 and idx < N:
                c[idx] = [x_start + col*s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.015, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 2. Corner-fitted dense patterns
    for seed in range(10):
        r_gen = np.random.RandomState(seed + 100)
        c = np.zeros((N, 2))
        idx = 0
        # Place circles in corners first
        corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
        for co in corners:
            c[idx] = co
            idx += 1
            
        # Fill rest with hex pattern
        s = 0.15 + r_gen.uniform(-0.02, 0.02)
        y = s
        row = 0
        while idx < N and y < 1.0 - s:
            x_start = s + (row % 2) * s/2
            col = 0
            while x_start + col*s < 1.0 - s and idx < N:
                c[idx] = [x_start + col*s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.15, 0.85, 2)
            idx += 1
        c += r_gen.normal(0, 0.01, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 3. Random uniform placements
    for _ in range(15):
        inits.append(rng.uniform(0.08, 0.92, (N, 2)))
        
    return inits

def hill_climb_centers(centers, rng, max_iter=60, init_step=0.025):
    """Coordinate-wise hill climbing on centers maximizing LP radii sum."""
    curr_c = centers.copy()
    curr_r = solve_lp_radii(curr_c)
    curr_sum = np.sum(curr_r)
    
    step = init_step
    for _ in range(max_iter):
        improved = False
        idxs = rng.permutation(N)
        for i in idxs:
            best_move = curr_c[i].copy()
            best_r = curr_r.copy()
            best_s = curr_sum
            
            # Try multiple directions for circle i
            for _ in range(6):
                direction = rng.normal(0, 1, 2)
                direction /= np.hypot(*direction)
                trial_c = curr_c.copy()
                trial_c[i] += direction * step
                trial_c[i] = np.clip(trial_c[i], 1e-5, 1.0 - 1e-5)
                
                trial_r = solve_lp_radii(trial_c)
                trial_s = np.sum(trial_r)
                
                if trial_s > best_s:
                    best_s = trial_s
                    best_move = trial_c[i].copy()
                    best_r = trial_r.copy()
            
            if best_s > curr_sum:
                curr_c[i] = best_move
                curr_r = best_r
                curr_sum = best_s
                improved = True
                
        if not improved:
            step *= 0.82
        if step < 1e-6:
            break
    return curr_c, curr_r, curr_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start SLSQP with LP initialization
    for c0 in inits:
        r0 = solve_lp_radii(c0) * 0.92
        r0 = np.maximum(r0, 1e-5)
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, 
                           options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                curr_c = np.column_stack((cx, cy))
                curr_r = solve_lp_radii(curr_c)
                curr_s = np.sum(curr_r)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            pass

    # Phase 2: Center-focused Hill Climbing & LP evaluation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for _ in range(20):
            hc_c, hc_r, hc_s = hill_climb_centers(curr_c, rng, max_iter=80, init_step=0.02)
            if hc_s > curr_s:
                curr_c, curr_r, curr_s = hc_c, hc_r, hc_s
                best_centers, best_radii, best_sum = hc_c.copy(), hc_r.copy(), hc_s
                
                # Polish after significant hill-climb improvement
                x0 = np.zeros(3*N)
                x0[0::3] = curr_c[:, 0]
                x0[1::3] = curr_c[:, 1]
                x0[2::3] = curr_r * 0.95
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, 
                                   options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        co = np.column_stack((res.x[0::3], res.x[1::3]))
                        ro = solve_lp_radii(co)
                        so = np.sum(ro)
                        if so > best_sum:
                            best_sum = so
                            best_centers = co.copy()
                            best_radii = ro.copy()
                            curr_c, curr_r, curr_s = co, ro, so
                except Exception:
                    pass

    # Phase 3: Deflate-Perturb-Expand Basin Hopping
    if best_centers is not None:
        for step_iter in range(40):
            scale = 0.015 * np.exp(-step_iter / 15.0)
            pert_c = best_centers + rng.normal(0, scale, best_centers.shape)
            pert_c = np.clip(pert_c, 0.01, 0.99)
            
            pert_r = solve_lp_radii(pert_c)
            pert_s = np.sum(pert_r)
            
            if pert_s > best_sum:
                best_sum = pert_s
                best_centers = pert_c.copy()
                best_radii = pert_r.copy()
                
                # Quick SLSQP polish on new basin
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = best_radii * 0.97
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, 
                                   options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        co = np.column_stack((res.x[0::3], res.x[1::3]))
                        ro = solve_lp_radii(co)
                        if np.sum(ro) > best_sum:
                            best_sum = np.sum(ro)
                            best_centers = co.copy()
                            best_radii = ro.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], max(0.0, mx - 1e-9))
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = math.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))