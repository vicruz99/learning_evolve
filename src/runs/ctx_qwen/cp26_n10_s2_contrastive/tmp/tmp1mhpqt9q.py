import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

A_ub_global = np.zeros((N_PAIRS, N))
A_ub_global[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_global[np.arange(N_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub_global, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(x):
    return -np.sum(x[2::3])

def constraints(x):
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c_dist = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    
    c_b = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    return np.concatenate([c_dist, c_b])

def push_apart(centers, radii, steps=20):
    """Force-directed separation of overlapping circles."""
    c = centers.copy()
    r = radii.copy()
    n = c.shape[0]
    for _ in range(steps):
        moves = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = math.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-12:
                    overlap = r[i] + r[j] - d
                    shift = overlap / 2.0
                    nx = dx / d
                    ny = dy / d
                    moves[i] += [nx * shift, ny * shift]
                    moves[j] -= [nx * shift, ny * shift]
        c += moves * 0.5
        c = np.clip(c, 0.001, 0.999)
    return c

def make_hex_init(spacing, seed, margin=0.04):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = margin
    while idx < N and y < 1.0 - margin:
        x_start = margin + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - margin and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_grid_init(spacing, seed, margin=0.04):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    y = margin
    while y < 1.0 - margin and idx < N:
        x = margin
        while x < 1.0 - margin and idx < N:
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            x += spacing
        y += spacing
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.006, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_corner_edge_init(seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    
    for cx, cy in [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]:
        centers[idx] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        idx += 1
    
    edge_pts = [
        (0.5, 0.08), (0.25, 0.08), (0.75, 0.08),
        (0.5, 0.92), (0.25, 0.92), (0.75, 0.92),
        (0.08, 0.5), (0.08, 0.25), (0.08, 0.75),
        (0.92, 0.5), (0.92, 0.25), (0.92, 0.75),
    ]
    for cx, cy in edge_pts:
        if idx < N:
            centers[idx] = [cx + rng.uniform(-0.015, 0.015), cy + rng.uniform(-0.015, 0.015)]
            idx += 1
    
    spacing = 0.14
    row = 0
    y = 0.3
    while idx < N and y < 0.7:
        x_start = 0.3 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 0.7 and idx < N:
            centers[idx, 0] = x_start + col * spacing
            centers[idx, 1] = y
            idx += 1
            col += 1
        y += spacing * np.sqrt(3) / 2
        row += 1
    
    while idx < N:
        centers[idx] = rng.uniform(0.2, 0.8, 2)
        idx += 1
    
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_row_pattern_init(seed, pattern=None):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    if pattern is None:
        pattern = [6, 5, 6, 5, 4]
    
    idx = 0
    y = 0.08
    dy = 0.84 / (len(pattern) - 0.5)
    
    for r_idx, cnt in enumerate(pattern):
        shift = 0.0 if r_idx % 2 == 0 else 0.085
        x = 0.06 + shift
        spacing = (0.88 - 2 * shift) / (cnt - 0.5) if cnt > 1 else 0.0
        for _ in range(cnt):
            if idx < N:
                centers[idx] = [x + rng.uniform(-0.005, 0.005), y + rng.uniform(-0.003, 0.003)]
                x += spacing
                idx += 1
        y += dy
    
    while idx < N:
        centers[idx] = rng.uniform(0.15, 0.85, 2)
        idx += 1
    
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal patterns
    for sp in np.linspace(0.14, 0.20, 12):
        for seed in range(6):
            inits.append(make_hex_init(sp, seed))
    
    # Grid patterns
    for sp in np.linspace(0.15, 0.20, 10):
        for seed in range(5):
            inits.append(make_grid_init(sp, seed))
    
    # Corner-edge patterns
    for seed in range(30):
        inits.append(make_corner_edge_init(seed))
    
    # Row patterns
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3]]
    for pat in patterns:
        for seed in range(10):
            inits.append(make_row_pattern_init(seed, pat))
    
    # Random
    for seed in range(30):
        rng_local = np.random.RandomState(seed + 500)
        inits.append(rng_local.uniform(0.08, 0.92, (N, 2)))
    
    # Phase 1: Initial optimization with relaxation
    for init_c in inits:
        r_init, s_init = solve_lp_radii(init_c)
        if s_init == 0:
            continue
        
        c_relaxed = push_apart(init_c, r_init, steps=20)
        r_relaxed, s_relaxed = solve_lp_radii(c_relaxed)
        
        x0 = np.zeros(3*N)
        x0[0::3] = c_relaxed[:, 0]
        x0[1::3] = c_relaxed[:, 1]
        x0[2::3] = np.maximum(r_relaxed * 0.98, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt, s_opt = solve_lp_radii(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Aggressive basin hopping
    if best_centers is not None:
        current_c = best_centers.copy()
        current_r = best_radii.copy()
        current_s = best_sum
        
        for step in range(200):
            noise_scale = 0.012 * np.exp(-step / 60.0)
            c_pert = current_c + rng.normal(0, noise_scale, current_c.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            c_relaxed = push_apart(c_pert, r_pert, steps=15)
            r_relaxed, s_relaxed = solve_lp_radii(c_relaxed)
            
            if s_relaxed > 0:
                x0 = np.zeros(3*N)
                x0[0::3] = c_relaxed[:, 0]
                x0[1::3] = c_relaxed[:, 1]
                x0[2::3] = r_relaxed
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                   options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    c_new = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_new, s_new = solve_lp_radii(c_new)
                    
                    if s_new > current_s:
                        current_s = s_new
                        current_c = c_new.copy()
                        current_r = r_new.copy()
                        if s_new > best_sum:
                            best_sum = s_new
                            best_centers = c_new.copy()
                            best_radii = r_new.copy()
                except Exception:
                    pass
    
    # Phase 3: Fine local search at multiple scales
    if best_centers is not None:
        for scale in [0.003, 0.001, 0.0005]:
            for _ in range(80):
                c_pert = best_centers + rng.normal(0, scale, best_centers.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                r_pert, s_pert = solve_lp_radii(c_pert)
                
                if s_pert > best_sum - 0.0005:
                    x0 = np.zeros(3*N)
                    x0[0::3] = c_pert[:, 0]
                    x0[1::3] = c_pert[:, 1]
                    x0[2::3] = np.maximum(r_pert * 0.99, 1e-5)
                    
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons_opt,
                                       options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_opt, s_opt = solve_lp_radii(c_opt)
                        if s_opt > best_sum:
                            best_sum = s_opt
                            best_centers = c_opt.copy()
                            best_radii = r_opt.copy()
                    except Exception:
                        pass
    
    # Fallback
    if best_centers is None:
        best_centers = make_hex_init(0.17, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # Strict post-processing
    radii = best_radii.copy()
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
    
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))