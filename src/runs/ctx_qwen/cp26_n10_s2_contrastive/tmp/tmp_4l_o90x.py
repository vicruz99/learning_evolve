import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Precompute constant matrix for LP pairwise constraints: r_i + r_j <= dist_ij
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
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

def generate_hex_init(rng, spacing, margin, noise_scale):
    """Generate hexagonal lattice initialization."""
    c = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2.0
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            c[idx, 0] = x_start + col * spacing
            c[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2.0
        row += 1
        
    while idx < N:
        c[idx] = rng.uniform(margin, 1.0 - margin, 2)
        idx += 1
        
    c += rng.normal(0, noise_scale, c.shape)
    return np.clip(c, 0.02, 0.98)

def generate_inits(rng):
    """Generate diverse structured initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying spacing, margins, and noise
    for seed in range(40):
        r_local = np.random.RandomState(seed)
        sp = 0.14 + r_local.uniform(0, 0.08)
        mg = 0.03 + r_local.uniform(0, 0.04)
        ns = r_local.uniform(0.005, 0.02)
        inits.append(generate_hex_init(rng, sp, mg, ns))
        
    # 2. Square grids
    for seed in range(20):
        r_local = np.random.RandomState(seed + 100)
        step = 0.16 + r_local.uniform(0, 0.06)
        mg = 0.04
        c = np.zeros((N, 2))
        idx = 0
        y = mg
        while idx < N and y < 1.0 - mg:
            x = mg
            while x < 1.0 - mg and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += step
            y += step
        while idx < N:
            c[idx] = r_local.uniform(mg, 1.0 - mg, 2)
            idx += 1
        c += r_local.normal(0, 0.01, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    # 3. Random uniform placements
    for _ in range(15):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def hill_climb_centers(centers, rng, steps=400, initial_step=0.012):
    """
    Black-box optimization on centers only, evaluating radii sum via LP.
    Uses adaptive step size and subset perturbations to escape local minima.
    """
    curr_c = centers.copy()
    r_curr = solve_lp_radii(curr_c)
    s_curr = np.sum(r_curr)
    step = initial_step
    
    for _ in range(steps):
        improved = False
        # Try multiple targeted perturbations per iteration
        for _ in range(40):
            pert = curr_c.copy()
            # Perturb a small random subset to reduce search dimensionality
            n_pert = rng.integers(1, 7)
            idx_pert = rng.choice(N, n_pert, replace=False)
            pert[idx_pert] += rng.normal(0, step, (n_pert, 2))
            pert = np.clip(pert, 0.01, 0.99)
            
            r_pert = solve_lp_radii(pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > s_curr:
                curr_c, r_curr, s_curr = pert, r_pert, s_pert
                improved = True
                
        if improved:
            step *= 1.08
            step = min(step, 0.04)
        else:
            step *= 0.85
        if step < 1e-6:
            break
            
    return curr_c, r_curr, s_curr

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    inits = generate_inits(rng)
    
    # Phase 1: Multi-start Joint SLSQP Optimization
    for c0 in inits:
        r0 = solve_lp_radii(c0) * 0.92
        r0 = np.maximum(r0, 1e-5)
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
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

    # Phase 2: LP-based Center Hill-Climbing to escape NLP local minima
    if best_centers is not None:
        # Run multiple independent hill-climbing trails from slightly perturbed best configs
        for trail in range(5):
            rng_t = np.random.default_rng(trail * 7 + 13)
            start_c = best_centers + rng_t.normal(0, 0.005, best_centers.shape)
            start_c = np.clip(start_c, 0.02, 0.98)
            
            hc_c, hc_r, hc_s = hill_climb_centers(start_c, rng_t, steps=500, initial_step=0.015)
            if hc_s > best_sum:
                best_sum = hc_s
                best_centers = hc_c.copy()
                best_radii = hc_r.copy()

    # Phase 3: Final Joint SLSQP Polish to tighten constraints and maximize radii
    if best_centers is not None:
        c_pol = best_centers.copy()
        r_pol = best_radii.copy()
        x0 = np.zeros(3 * N)
        x0[0::3] = c_pol[:, 0]
        x0[1::3] = c_pol[:, 1]
        x0[2::3] = r_pol * 0.98
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_final = np.column_stack((res.x[0::3], res.x[1::3]))
                r_final = solve_lp_radii(c_final)
                s_final = np.sum(r_final)
                if s_final > best_sum:
                    best_sum = s_final
                    best_centers = c_final
                    best_radii = r_final
        except Exception:
            pass

    # Phase 4: Strict post-processing to guarantee validator compliance
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        best_radii[i] = min(best_radii[i], max(0.0, mx - 1e-9))
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(100):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            if d < best_radii[i] + best_radii[j] - 1e-11:
                exc = best_radii[i] + best_radii[j] - d
                best_radii[i] -= exc * 0.5
                best_radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
            
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))