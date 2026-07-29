import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0


def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, mx)))
    
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    
    try:
        res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    
    return np.full(n, 0.01), 0.26


def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])


def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
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


def generate_hex_init(rng, spacing, margin=0.04, noise_scale=0.006):
    """Generate hexagonal lattice initialization."""
    c = np.zeros((N, 2))
    idx = 0
    y = margin
    row = 0
    while idx < N and y < 1.0 - margin:
        x = margin + (row % 2) * spacing / 2.0
        while x < 1.0 - margin and idx < N:
            c[idx, 0] = x + rng.uniform(-noise_scale, noise_scale)
            c[idx, 1] = y + rng.uniform(-noise_scale, noise_scale)
            idx += 1
            x += spacing
        y += spacing * math.sqrt(3) / 2.0
        row += 1
    
    while idx < N:
        c[idx] = rng.uniform(margin + 0.05, 1.0 - margin - 0.05, 2)
        idx += 1
    
    return np.clip(c, 0.01, 0.99)


def generate_square_grid_init(rng, step, margin=0.04, noise_scale=0.005):
    """Generate square grid initialization."""
    c = np.zeros((N, 2))
    idx = 0
    y = margin
    while y < 1.0 - margin and idx < N:
        x = margin
        while x < 1.0 - margin and idx < N:
            c[idx, 0] = x + rng.uniform(-noise_scale, noise_scale)
            c[idx, 1] = y + rng.uniform(-noise_scale, noise_scale)
            idx += 1
            x += step
        y += step
    
    while idx < N:
        c[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    
    return np.clip(c, 0.01, 0.99)


def generate_row_pattern_init(rng, pattern, noise_scale=0.006):
    """Generate centers based on row pattern with hexagonal offsets."""
    c = np.zeros((N, 2))
    idx = 0
    n_rows = len(pattern)
    y_start = 0.03
    y_end = 0.97
    dy = (y_end - y_start) / max(n_rows - 1, 1) if n_rows > 1 else 0.0
    
    for row_i, cnt in enumerate(pattern):
        y = y_start + row_i * dy
        shift = 0.0 if row_i % 2 == 0 else 0.06
        x_start = 0.03 + shift
        total_width = 0.94 - 2 * shift
        x_step = total_width / max(cnt - 1, 1) if cnt > 1 else 0.0
        
        for j in range(cnt):
            if idx < N:
                c[idx, 0] = x_start + j * x_step + rng.uniform(-noise_scale, noise_scale)
                c[idx, 1] = y + rng.uniform(-noise_scale, noise_scale)
                idx += 1
    
    while idx < N:
        c[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    
    return np.clip(c, 0.01, 0.99)


def generate_corner_focused_init(rng, noise_scale=0.008):
    """Generate initialization with circles near corners and edges."""
    c = np.zeros((N, 2))
    idx = 0
    margin = 0.06
    
    # Place circles in corners
    corners = [[margin, margin], [1-margin, margin], [margin, 1-margin], [1-margin, 1-margin]]
    for corner in corners:
        if idx < N:
            c[idx] = np.array(corner) + rng.uniform(-noise_scale, noise_scale, 2)
            idx += 1
    
    # Place circles along edges
    edge_points = []
    for k in range(5):
        t = 0.25 * (k + 0.5)
        edge_points.extend([
            [t, margin], [1-t, margin],
            [t, 1-margin], [1-t, 1-margin],
            [margin, t], [margin, 1-t],
            [1-margin, t], [1-margin, 1-t]
        ])
    
    rng.shuffle(edge_points)
    for pt in edge_points:
        if idx < N:
            c[idx] = np.array(pt) + rng.uniform(-noise_scale, noise_scale, 2)
            idx += 1
    
    # Fill remaining in center
    while idx < N:
        c[idx] = rng.uniform(0.3, 0.7, 2) + rng.uniform(-noise_scale, noise_scale, 2)
        idx += 1
    
    return np.clip(c, 0.01, 0.99)


def run_slqp_from(c0, r0, rng, maxiter=10000):
    """Run SLSQP optimization from given initial configuration."""
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    x0 = np.zeros(3 * N)
    x0[0::3] = c0[:, 0]
    x0[1::3] = c0[:, 1]
    x0[2::3] = np.maximum(r0 * 0.95, 1e-5)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False})
        if res.success:
            cx = res.x[0::3]
            cy = res.x[1::3]
            c_opt = np.column_stack((cx, cy))
            r_opt, s_opt = solve_lp_radii(c_opt)
            return c_opt, r_opt, s_opt
    except Exception:
        pass
    return c0, r0, np.sum(r0)


def basin_hopping_centers(current_c, current_s, rng, n_steps=200, step_scale=0.008):
    """Iterative basin hopping on centers with LP radii evaluation."""
    c = current_c.copy()
    s = current_s
    
    for step in range(n_steps):
        scale = step_scale * (0.95 ** (step // 20))
        
        # Perturb subset of circles
        n_pert = rng.integers(2, min(7, N))
        idx_pert = rng.choice(N, n_pert, replace=False)
        
        c_pert = c.copy()
        c_pert[idx_pert] += rng.normal(0, scale, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        
        r_pert, s_pert = solve_lp_radii(c_pert)
        
        if s_pert > s:
            c = c_pert
            s = s_pert
    
    return c, s


def refine_with_momentum(current_c, rng, n_iter=100):
    """Refine using momentum-based search on promising directions."""
    c = current_c.copy()
    r, s_best = solve_lp_radii(c)
    
    best_c = c.copy()
    best_s = s_best
    
    step_size = 0.005
    for it in range(n_iter):
        improved = False
        for _ in range(30):
            n_pert = rng.integers(1, 5)
            idx_pert = rng.choice(N, n_pert, replace=False)
            
            c_try = c.copy()
            c_try[idx_pert] += rng.normal(0, step_size, (n_pert, 2))
            c_try = np.clip(c_try, 0.01, 0.99)
            
            r_try, s_try = solve_lp_radii(c_try)
            
            if s_try > best_s:
                c = c_try
                best_c = c_try
                best_s = s_try
                improved = True
        
        if not improved:
            step_size *= 0.85
        else:
            step_size = min(step_size * 1.05, 0.02)
        
        if step_size < 1e-6:
            break
    
    return best_c, best_s


def make_strictly_valid(centers, radii):
    """Ensure configuration strictly satisfies all constraints."""
    c = centers.copy()
    r = radii.copy()
    
    # Enforce boundary constraints
    for i in range(N):
        mx = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        r[i] = min(r[i], max(0.0, mx - 1e-9))
    
    # Iteratively fix overlaps
    for _ in range(80):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = math.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            if d < r[i] + r[j] - 1e-12:
                exc = r[i] + r[j] - d
                r[i] -= exc * 0.5
                r[j] -= exc * 0.5
                changed = True
        if not changed:
            break
    
    r = np.maximum(r, 0.0)
    return c, r


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing in unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # ---- Phase 1: Generate diverse initial configurations ----
    inits = []
    
    # Hexagonal lattices with various spacings
    for seed in range(25):
        r_gen = np.random.RandomState(seed * 13 + 1)
        spacing = 0.15 + r_gen.uniform(-0.02, 0.06)
        margin = 0.03 + r_gen.uniform(-0.01, 0.02)
        c = generate_hex_init(r_gen, spacing, margin, 0.007)
        inits.append(c)
    
    # Square grids
    for seed in range(15):
        r_gen = np.random.RandomState(seed * 17 + 3)
        step = 0.17 + r_gen.uniform(-0.02, 0.04)
        c = generate_square_grid_init(r_gen, step, 0.04, 0.006)
        inits.append(c)
    
    # Row patterns
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6], [8, 6, 5, 4, 3], [6, 6, 6, 6, 2],
        [5, 5, 5, 5, 6], [7, 5, 5, 5, 4], [6, 5, 5, 5, 5],
        [4, 6, 6, 6, 4], [5, 5, 6, 5, 5]
    ]
    for pat in patterns:
        c = generate_row_pattern_init(rng, pat, 0.007)
        inits.append(c)
    
    # Corner-focused
    for seed in range(15):
        r_gen = np.random.RandomState(seed * 19 + 7)
        c = generate_corner_focused_init(r_gen, 0.008)
        inits.append(c)
    
    # Pure random
    for seed in range(20):
        r_gen = np.random.RandomState(seed * 23 + 11)
        c = r_gen.uniform(0.08, 0.92, (N, 2))
        inits.append(c)
    
    # ---- Phase 2: Optimize each initialization ----
    for c0 in inits:
        r0, s0 = solve_lp_radii(c0)
        c_opt, r_opt, s_opt = run_slqp_from(c0, r0, rng, maxiter=10000)
        
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    
    # ---- Phase 3: Basin hopping from best ----
    if best_centers is not None:
        c_bh, s_bh = basin_hopping_centers(best_centers, best_sum, rng, n_steps=300, step_scale=0.01)
        if s_bh > best_sum:
            best_sum = s_bh
            best_centers = c_bh
            best_radii, _ = solve_lp_radii(c_bh)
    
    # ---- Phase 4: Multiple refinement passes with decreasing step sizes ----
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_s = best_sum
        
        for scale in [0.006, 0.004, 0.002, 0.001]:
            c_ref, s_ref = refine_with_momentum(curr_c, rng, n_iter=80)
            if s_ref > curr_s:
                curr_c = c_ref
                curr_s = s_ref
                best_sum = s_ref
                best_centers = c_ref
                best_radii, _ = solve_lp_radii(c_ref)
        
        # Final SLSQP polish
        r_lp, _ = solve_lp_radii(best_centers)
        c_polish, r_polish, s_polish = run_slqp_from(best_centers, r_lp, rng, maxiter=12000)
        if s_polish > best_sum:
            best_sum = s_polish
            best_centers = c_polish
            best_radii = r_polish
    
    # ---- Phase 5: Additional multi-start from best with perturbation ----
    if best_centers is not None:
        for trial in range(30):
            c_pert = best_centers.copy()
            scale = 0.003 * (1.0 + trial * 0.15)
            n_pert = rng.integers(3, 10)
            idx_pert = rng.choice(N, n_pert, replace=False)
            c_pert[idx_pert] += rng.normal(0, scale, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, _ = solve_lp_radii(c_pert)
            c_opt, r_opt, s_opt = run_slqp_from(c_pert, r_pert, rng, maxiter=6000)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = generate_hex_init(rng, 0.18)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # ---- Phase 6: Strict post-processing ----
    best_centers, best_radii = make_strictly_valid(best_centers, best_radii)
    
    return best_centers, best_radii, float(np.sum(best_radii))