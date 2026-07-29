import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    c_obj = -np.ones(N)
    A_ub = np.zeros((NUM_PAIRS, N))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 1e-5)

def relax_centers(centers, radii, steps=10):
    """Deterministically push overlapping circles apart."""
    c = centers.copy()
    r = radii.copy()
    for _ in range(steps):
        forces = np.zeros_like(c)
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = np.hypot(dx, dy)
            if d < r[i] + r[j] and d > 1e-8:
                overlap = r[i] + r[j] - d
                shift = overlap / (2.0 * d)
                forces[i] += np.array([dx * shift, dy * shift])
                forces[j] -= np.array([dx * shift, dy * shift])
        c += forces * 0.2
        c = np.clip(c, 1e-5, 1.0 - 1e-5)
    return c

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
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

def hill_climb(centers, rng, max_iter=120):
    """Optimize centers by perturbing, relaxing, and evaluating LP radii sum."""
    curr_c = centers.copy()
    curr_r = solve_lp_radii(curr_c)
    curr_s = np.sum(curr_r)
    step = 0.012
    
    for _ in range(max_iter):
        improved = False
        for _ in range(25):
            n_pert = rng.integers(1, 5)
            idx = rng.choice(N, n_pert, replace=False)
            p_c = curr_c.copy()
            p_c[idx] += rng.normal(0, step, (n_pert, 2))
            p_c = np.clip(p_c, 0.001, 0.999)
            
            # Relax to resolve tight overlaps before LP evaluation
            p_c_relaxed = relax_centers(p_c, curr_r, steps=8)
            p_r = solve_lp_radii(p_c_relaxed)
            p_s = np.sum(p_r)
            
            if p_s > curr_s:
                curr_c, curr_r, curr_s = p_c_relaxed, p_r, p_s
                improved = True
                
        if improved:
            step = min(step * 1.08, 0.025)
        else:
            step *= 0.88
            if step < 1e-6:
                break
    return curr_c, curr_r, curr_s

def generate_inits(rng):
    """Generate diverse structured and random initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying spacings
    for seed in range(25):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        s = 0.14 + r_gen.uniform(0, 0.06)
        y = s / 2
        row = 0
        while idx < N and y < 1.0 - s / 2:
            x_start = s / 2 + (row % 2) * s / 2
            col = 0
            while x_start + col * s < 1.0 - s / 2 and idx < N:
                c[idx] = [x_start + col * s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.008, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 2. Square grids
    for seed in range(15):
        r_gen = np.random.RandomState(seed + 100)
        c = np.zeros((N, 2))
        idx = 0
        s = 0.16 + r_gen.uniform(-0.02, 0.03)
        y = s / 2
        while y < 1.0 - s / 2 and idx < N:
            x = s / 2
            while x < 1.0 - s / 2 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += s
            y += s
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.005, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 3. Corner-biased random patterns
    for seed in range(20):
        r_gen = np.random.RandomState(seed + 200)
        c = r_gen.uniform(0.05, 0.95, (N, 2))
        # Push some towards corners
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        for i in range(4):
            c[i] = corners[i] + r_gen.uniform(-0.02, 0.02, 2)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 4. Pure random
    for _ in range(15):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start Hill-Climbing & SLSQP Polish
    for c0 in inits:
        r0 = solve_lp_radii(c0)
        c_hc, r_hc, s_hc = hill_climb(c0, rng, max_iter=100)
        
        # SLSQP Polish
        x0 = np.zeros(3 * N)
        x0[0::3] = c_hc[:, 0]
        x0[1::3] = c_hc[:, 1]
        x0[2::3] = np.maximum(r_hc * 0.95, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                c_opt = np.column_stack((cx, cy))
                r_opt = solve_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            if s_hc > best_sum:
                best_sum = s_hc
                best_centers = c_hc.copy()
                best_radii = r_hc.copy()

    # Phase 2: Aggressive Basin Hopping on Best Configuration
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step_iter in range(60):
            scale = 0.008 * np.exp(-step_iter / 25.0)
            for _ in range(20):
                p_c = curr_c + rng.normal(0, scale, (N, 2))
                p_c = np.clip(p_c, 0.01, 0.99)
                p_r = solve_lp_radii(relax_centers(p_c, curr_r, steps=5))
                p_s = np.sum(p_r)
                if p_s > curr_s:
                    curr_c, curr_r, curr_s = p_c, p_r, p_s
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
                        
                        # Quick SLSQP polish on new best
                        x0 = np.zeros(3 * N)
                        x0[0::3] = best_centers[:, 0]
                        x0[1::3] = best_centers[:, 1]
                        x0[2::3] = np.maximum(best_radii * 0.96, 1e-5)
                        try:
                            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                           constraints=cons_opt, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                            if res.success:
                                c_p = np.column_stack((res.x[0::3], res.x[1::3]))
                                r_p = solve_lp_radii(c_p)
                                s_p = np.sum(r_p)
                                if s_p > best_sum:
                                    best_sum = s_p
                                    best_centers = c_p.copy()
                                    best_radii = r_p.copy()
                        except Exception:
                            pass

    # Phase 3: Strict post-processing to guarantee validator compliance
    if best_centers is not None:
        radii = best_radii.copy()
        
        # Enforce boundary constraints strictly
        for i in range(N):
            mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                     best_centers[i, 1], 1.0 - best_centers[i, 1])
            radii[i] = min(radii[i], max(0.0, mx - 1e-9))
            
        # Iteratively resolve any remaining numerical overlaps
        for _ in range(100):
            changed = False
            for k in range(NUM_PAIRS):
                i, j = I_IDX[k], J_IDX[k]
                d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-11:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc * 0.5
                    radii[j] -= exc * 0.5
                    changed = True
            if not changed:
                break
                
        radii = np.maximum(radii, 0.0)
        return best_centers, radii, float(np.sum(radii))
        
    # Fallback safety net
    c_fallback = np.random.uniform(0.1, 0.9, (N, 2))
    r_fallback = solve_lp_radii(c_fallback)
    return c_fallback, r_fallback, float(np.sum(r_fallback))