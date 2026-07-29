import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
N_PAIRS = len(I_IDX)

# Precompute LP constraint matrix structure for speed
A_ub_lp = np.zeros((N_PAIRS, N))
A_ub_lp[np.arange(N_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(N_PAIRS), J_IDX] = 1.0

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

def solve_lp_radii(centers):
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(0.0, mx)))
    
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(N), 0.0

def polish(c_init):
    r_init, s_init = solve_lp_radii(c_init)
    if s_init <= 0:
        return c_init, np.zeros(N), 0.0
    
    r_init = np.maximum(r_init * 0.97, 1e-5)
    x0 = np.zeros(3 * N)
    x0[0::3] = c_init[:, 0]
    x0[1::3] = c_init[:, 1]
    x0[2::3] = r_init
    
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                       constraints=cons_opt,
                       options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
        c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
        r_opt, s_opt = solve_lp_radii(c_opt)
        return c_opt, r_opt, s_opt
    except Exception:
        return c_init, r_init, s_init

def make_hex(spacing, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    row = 0
    y = spacing / 2
    while idx < N and y < 1.0 - spacing / 2:
        x_start = spacing / 2 + (row % 2) * spacing / 2
        col = 0
        while x_start + col * spacing < 1.0 - spacing / 2 and idx < N:
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

def make_grid(spacing, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    y = spacing / 2
    while y < 1.0 - spacing / 2 and idx < N:
        x = spacing / 2
        while x < 1.0 - spacing / 2 and idx < N:
            centers[idx] = [x, y]
            x += spacing
            idx += 1
        y += spacing
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.006, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_staggered(rows_pattern, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    n_rows = len(rows_pattern)
    y = 0.05
    dy = 0.9 / n_rows
    for r_idx, cnt in enumerate(rows_pattern):
        shift = 0.0 if r_idx % 2 == 0 else 0.09
        x = 0.05 + shift
        for _ in range(cnt):
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
            if cnt > 1:
                x += (0.88 - 2 * shift) / (cnt - 1)
        y += dy
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, 0.02, 0.98)

def make_corner_biased(seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    idx = 0
    
    # Four corner regions
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
    for c in corners:
        if idx < N:
            centers[idx] = [c[0] + rng.uniform(-0.02, 0.02), c[1] + rng.uniform(-0.02, 0.02)]
            idx += 1
    
    # Edge midpoints
    edges = [[0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5]]
    for e in edges:
        if idx < N:
            centers[idx] = [e[0] + rng.uniform(-0.03, 0.03), e[1] + rng.uniform(-0.03, 0.03)]
            idx += 1
    
    # Fill rest in a hex pattern
    sp = 0.17
    row = 0
    y = sp / 2
    while idx < N and y < 1.0 - sp / 2:
        x_start = sp / 2 + (row % 2) * sp / 2
        col = 0
        while x_start + col * sp < 1.0 - sp / 2 and idx < N:
            cx = x_start + col * sp
            cy = y
            # Check if this position is far from existing centers
            too_close = False
            for k in range(idx):
                d = math.hypot(cx - centers[k, 0], cy - centers[k, 1])
                if d < 0.06:
                    too_close = True
                    break
            if not too_close:
                centers[idx] = [cx, cy]
                idx += 1
            col += 1
        y += sp * np.sqrt(3) / 2
        row += 1
    
    while idx < N:
        centers[idx] = rng.uniform(0.1, 0.9, 2)
        idx += 1
    
    return np.clip(centers, 0.03, 0.97)

def directional_refine(centers, radii, rng, steps=30):
    """Try moving each circle in 8 compass directions to improve sum of radii."""
    c = centers.copy()
    r = radii.copy()
    s = np.sum(r)
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ])
    directions /= np.sqrt(2)
    
    for _ in range(steps):
        improved = False
        for i in range(N):
            best_s = s
            best_c = c[i].copy()
            
            for scale in [0.005, 0.01, 0.02]:
                for d in directions:
                    c_trial = c.copy()
                    c_trial[i] = c[i] + d * scale
                    c_trial[i] = np.clip(c_trial[i], 0.02, 0.98)
                    
                    r_trial, s_trial = solve_lp_radii(c_trial)
                    
                    if s_trial > best_s + 1e-12:
                        best_s = s_trial
                        best_c = c_trial[i].copy()
            
            if best_s > s + 1e-12:
                c[i] = best_c
                c_new = c.copy()
                r, s = solve_lp_radii(c)
                c = c_new
                improved = True
        
        if not improved:
            break
    
    return c, r, s

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(42)
    
    configs = []
    
    # Hexagonal with various spacings
    for sp in np.linspace(0.11, 0.24, 30):
        for seed in range(10):
            configs.append(make_hex(sp, seed))
    
    # Grid
    for sp in np.linspace(0.13, 0.25, 18):
        for seed in range(8):
            configs.append(make_grid(sp, seed))
    
    # Staggered patterns
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6],
        [8,6,5,4,3], [6,6,6,6,2], [5,5,5,5,6], [6,5,5,5,5],
        [7,5,6,5,3], [5,7,5,6,3], [6,4,6,6,4], [4,7,7,5,3],
        [7,6,6,4,3], [5,6,6,6,3], [6,6,5,5,4], [4,5,7,5,5],
        [8,5,5,5,3], [5,7,6,4,4], [7,7,5,5,2], [6,5,7,4,4]
    ]
    for pat in patterns:
        for seed in range(8):
            configs.append(make_staggered(pat, seed))
    
    # Corner-biased configurations
    for seed in range(20):
        configs.append(make_corner_biased(seed))
    
    # Random
    for seed in range(80):
        configs.append(rng.uniform(0.08, 0.92, (N, 2)))
    
    # Phase 1: Polish all configs
    for c_init in configs:
        c_opt, r_opt, s_opt = polish(c_init)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    
    # Phase 2: Directional refinement on best
    if best_centers is not None:
        c_ref, r_ref, s_ref = directional_refine(best_centers, best_radii, rng, steps=20)
        if s_ref > best_sum:
            best_sum = s_ref
            best_centers = c_ref.copy()
            best_radii = r_ref.copy()
            c_pol, r_pol, s_pol = polish(best_centers)
            if s_pol > best_sum:
                best_sum = s_pol
                best_centers = c_pol.copy()
                best_radii = r_pol.copy()
    
    # Phase 3: Simulated annealing basin hopping (aggressive)
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(500):
            # Temperature schedule: slow decay
            temperature = 0.03 * np.exp(-step / 120.0)
            noise_scale = 0.012 * np.sqrt(max(temperature / 0.03, 0.0005))
            
            c_pert = curr_c + rng.normal(0, noise_scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            delta = s_pert - curr_s
            accept = delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-8))
            
            if accept and s_pert > 0:
                curr_c = c_pert
                curr_r = r_pert
                curr_s = s_pert
                
                # Polish at intervals or on improvement
                if step % 3 == 0 or s_pert > best_sum:
                    c_pol, r_pol, s_pol = polish(curr_c)
                    if s_pol > curr_s:
                        curr_c = c_pol
                        curr_r = r_pol
                        curr_s = s_pol
                    
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
                        
                        # Directional refinement after new best
                        if step % 20 == 0:
                            c_ref, r_ref, s_ref = directional_refine(best_centers, best_radii, rng, steps=10)
                            if s_ref > best_sum:
                                best_sum = s_ref
                                best_centers = c_ref.copy()
                                best_radii = r_ref.copy()
    
    # Phase 4: Additional basin hopping with different noise patterns
    if best_centers is not None:
        for trial in range(30):
            # Different noise: perturb 1-4 random circles more aggressively
            num_pert = rng.choice([1, 2, 3, 4])
            idxs = rng.choice(N, num_pert, replace=False)
            
            c_pert = best_centers.copy()
            for idx in idxs:
                c_pert[idx] += rng.normal(0, 0.02, 2)
                c_pert[idx] = np.clip(c_pert[idx], 0.03, 0.97)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum + 1e-12:
                c_pol, r_pol, s_pol = polish(c_pert)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
    
    # Phase 5: Coordinate descent with random restarts
    if best_centers is not None:
        for restart in range(5):
            c_cur = best_centers.copy() + rng.normal(0, 0.003, best_centers.shape)
            c_cur = np.clip(c_cur, 0.02, 0.98)
            r_cur, s_cur = solve_lp_radii(c_cur)
            
            for iteration in range(50):
                improved = False
                for i in range(N):
                    for scale in [0.003, 0.008, 0.015]:
                        for _ in range(6):
                            c_trial = c_cur.copy()
                            c_trial[i] += rng.normal(0, scale, 2)
                            c_trial[i] = np.clip(c_trial[i], 0.02, 0.98)
                            r_trial, s_trial = solve_lp_radii(c_trial)
                            if s_trial > s_cur + 1e-12:
                                s_cur = s_trial
                                r_cur = r_trial.copy()
                                c_cur = c_trial.copy()
                                improved = True
                if not improved:
                    break
            
            if s_cur > best_sum + 1e-12:
                best_sum = s_cur
                best_centers = c_cur.copy()
                best_radii = r_cur.copy()
                c_pol, r_pol, s_pol = polish(best_centers)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
    
    # Phase 6: Final fine-tuning
    if best_centers is not None:
        for _ in range(30):
            c_fine = best_centers + rng.normal(0, 0.0008, best_centers.shape)
            c_fine = np.clip(c_fine, 0.01, 0.99)
            c_pol, r_pol, s_pol = polish(c_fine)
            if s_pol > best_sum:
                best_sum = s_pol
                best_centers = c_pol.copy()
                best_radii = r_pol.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = make_hex(0.17, 0)
        best_radii, best_sum = solve_lp_radii(best_centers)
    
    # Strict post-processing to guarantee validator compliance
    radii = best_radii.copy()
    
    for i in range(N):
        mx = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                 best_centers[i, 1], 1.0 - best_centers[i, 1])
        radii[i] = min(radii[i], mx - 1e-9)
        radii[i] = max(0.0, radii[i])
    
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(best_centers[i, 0] - best_centers[j, 0],
                               best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))