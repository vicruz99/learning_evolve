import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant structure for LP pairwise constraints
A_LP = np.zeros((NUM_PAIRS, N))
A_LP[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_LP[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for x, y in centers:
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

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
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def generate_inits(rng):
    """Generate diverse initial center configurations."""
    inits = []
    # 1. Hexagonal lattices with varying spacings and offsets
    for sp in np.linspace(0.16, 0.26, 10):
        for shift_y in [0.0, 0.04, 0.08]:
            c = np.zeros((N, 2))
            idx = 0
            y = 0.04 + shift_y
            row = 0
            while idx < N and y < 0.96:
                x = 0.04 + (row % 2) * sp / 2.0
                while x < 0.96 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += sp
                y += sp * np.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(c + rng.normal(0, 0.005, c.shape))
            
    # 2. Square grids
    for step in np.linspace(0.16, 0.24, 8):
        c = np.zeros((N, 2))
        idx = 0
        y = 0.05
        while y < 0.95 and idx < N:
            x = 0.05
            while x < 0.95 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += step
            y += step
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.004, c.shape))
        
    # 3. Corner-focused pattern
    c_corners = np.array([[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]])
    c_mid = rng.uniform(0.2, 0.8, (N - 4, 2))
    c_full = np.vstack([c_corners, c_mid])
    inits.append(c_full + rng.normal(0, 0.005, c_full.shape))
        
    # 4. Random uniform placements
    for _ in range(30):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def jiggle_refine(centers, radii, steps=3000, rng=None):
    """Hill-climbing local search on centers to maximize LP sum of radii."""
    if rng is None:
        rng = np.random.default_rng()
        
    c = centers.copy()
    r = radii.copy()
    current_sum = np.sum(r)
    step = 0.008
    
    for _ in range(steps):
        step *= 0.9992  # Slow decay
        if step < 1e-5:
            break
            
        # Perturb 1 to 3 random circles
        n_pert = rng.integers(1, 4)
        idxs = rng.choice(N, n_pert, replace=False)
        c_pert = c.copy()
        c_pert[idxs] += rng.normal(0, step, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.005, 0.995)
        
        r_pert = solve_lp_radii(c_pert)
        s_pert = np.sum(r_pert)
        
        if s_pert > current_sum:
            c, r, current_sum = c_pert, r_pert, s_pert
            
    return c, r, current_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start SLSQP
    for base in inits:
        c_init = np.clip(base, 0.02, 0.98)
        r_init = solve_lp_radii(c_init) * 0.97
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = np.maximum(r_init, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt = solve_lp_radii(c_opt)
            s_opt = np.sum(r_opt)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Intensive Jiggle + SLSQP Hybrid Refinement
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        # Run multiple refinement cycles
        for cycle in range(6):
            # Jiggle refinement
            curr_c, curr_r, curr_s = jiggle_refine(curr_c, curr_r, steps=2500, rng=rng)
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
            # SLSQP polish after jiggle
            x0_p = np.zeros(3 * N)
            x0_p[0::3] = curr_c[:, 0]
            x0_p[1::3] = curr_c[:, 1]
            x0_p[2::3] = np.maximum(curr_r * 0.98, 1e-5)
            
            try:
                res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                 constraints=cons_opt,
                                 options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                c_pol = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                r_pol = solve_lp_radii(c_pol)
                s_pol = np.sum(r_pol)
                
                if s_pol > curr_s:
                    curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
            except Exception:
                pass

    # Phase 3: Final fine-tuning jiggle
    if best_centers is not None:
        best_centers, best_radii, best_sum = jiggle_refine(best_centers, best_radii, steps=1500, rng=rng)
        
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
        ub = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(150):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))