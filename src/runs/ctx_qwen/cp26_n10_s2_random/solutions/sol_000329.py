# sol_000329 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000297 (state 4b6f1fd1) state=cd80e7e7 sum of radii=2.577390 correctness=1.0
# stdout(first 200): Testing 299 initial configurations... Processing start 0/299, best so far: -1.000000   New best: 2.577390 Processing start 10/299, best so far: 2.577390 Processing start 20/299, best so far: 2.577390 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[idx, i] = 1.0
        A_LP[idx, j] = 1.0
        PAIR_IDX.append((i, j))
        idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii and computes exact subgradient via duals."""
    centers = np.clip(centers, 1e-12, 1.0 - 1e-12)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]
        k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += mu * vec
                grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
    return radii, s_sum, grad

def ga_optimize_line_search(centers0, max_iter=4000):
    """Gradient ascent with backtracking line search for precision."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = 0.015
    
    r, s, g = solve_lp_and_grad(centers)
    best_s = s
    
    for iteration in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
            
        g_dir = g / gn
        
        # Backtracking line search
        best_step = 0.0
        best_ns = s
        for trial in range(15):
            step_try = step * (0.5 ** trial)
            if step_try < 1e-12:
                break
            nc = np.clip(centers + step_try * g_dir, 1e-6, 1.0 - 1e-6)
            _, ns, _ = solve_lp_and_grad(nc)
            if ns > best_ns:
                best_ns = ns
                best_step = step_try
        
        if best_step > 0 and best_ns > s:
            centers = np.clip(centers + best_step * g_dir, 1e-6, 1.0 - 1e-6)
            s = best_ns
            r, s, g = solve_lp_and_grad(centers)
            if s > best_s:
                best_s = s
                best_c = centers.copy()
            step = min(step * 1.2, 0.05)
        else:
            step *= 0.4
            if step < 1e-13:
                break
        
        # Periodic jitter
        if iteration > 0 and iteration % 500 == 0:
            noise = np.random.normal(0, 0.003, centers.shape)
            centers = np.clip(centers + noise, 1e-6, 1.0 - 1e-6)
            r, s, g = solve_lp_and_grad(centers)
    
    return best_c, best_s

def ga_optimize_simple(centers0, max_iter=3000, step_init=0.01):
    """Gradient ascent with adaptive step size."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = step_init
    
    r, s, g = solve_lp_and_grad(centers)
    best_s = s
    
    for iteration in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
            
        g_dir = g / gn
        nc = np.clip(centers + step * g_dir, 1e-6, 1.0 - 1e-6)
        _, ns, _ = solve_lp_and_grad(nc)
        
        if ns > s + 1e-14:
            centers = nc
            s = ns
            r, s, g = solve_lp_and_grad(centers)
            step = min(step * 1.1, 0.04)
            if s > best_s:
                best_s = s
                best_c = centers.copy()
        else:
            step *= 0.6
            if step < 1e-13:
                break
    
    return best_c, best_s

def obj_joint(v):
    return -np.sum(v[2*N:])

def cons_joint(v):
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = []
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def slsqp_polish(c0, r0, max_iter=20000):
    """Joint SLSQP optimization with extended iterations."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False})
        c_val = cons_joint(res.x)
        if np.min(c_val) >= -1e-6:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_hex_pattern(pat, r0, rng):
    """Generate a hexagonal lattice pattern."""
    c = []
    y = r0
    for ri, cnt in enumerate(pat):
        sh = r0 if ri % 2 == 1 else 0.0
        x = r0 + sh
        for _ in range(cnt):
            if len(c) < N:
                c.append([x + rng.normal(0, 0.001), y + rng.normal(0, 0.001)])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
    while len(c) < N:
        c.append([0.5 + rng.normal(0, 0.1), 0.5 + rng.normal(0, 0.1)])
    return np.clip(np.array(c[:N]), 0.05, 0.95)

def generate_corner_optimized(rng):
    """Generate configurations optimized for corner utilization."""
    starts = []
    # Place 4 circles in corners, then fill with hex pattern
    for _ in range(6):
        c = np.zeros((N, 2))
        corner_offset = 0.12 + rng.uniform(0, 0.03)
        c[0] = [corner_offset, corner_offset]
        c[1] = [1.0 - corner_offset, corner_offset]
        c[2] = [corner_offset, 1.0 - corner_offset]
        c[3] = [1.0 - corner_offset, 1.0 - corner_offset]
        
        # Fill remaining with hex-like pattern in center
        remaining = N - 4
        pts = []
        r_est = 0.09
        y = 0.3
        for row in range(6):
            shift = r_est if row % 2 == 1 else 0.0
            x = 0.3 + shift
            while x <= 0.7 and len(pts) < remaining:
                pts.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
                x += 2.0 * r_est
            y += r_est * np.sqrt(3.0)
        
        c[4:4+len(pts)] = pts[:remaining]
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
    return starts

def generate_rotated_hex(rng):
    """Generate rotated hexagonal patterns."""
    starts = []
    for angle_deg in [0, 15, 30, 45]:
        angle = np.radians(angle_deg)
        pat = [5, 6, 5, 6, 4]
        for r0 in [0.09, 0.095, 0.10]:
            c = generate_hex_pattern(pat, r0, rng)
            # Rotate around center
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            centered = c - 0.5
            rotated = np.column_stack([
                centered[:, 0] * cos_a - centered[:, 1] * sin_a,
                centered[:, 0] * sin_a + centered[:, 1] * cos_a
            ])
            c_rot = rotated + 0.5
            c_rot = np.clip(c_rot, 0.05, 0.95)
            starts.append(c_rot)
    return starts

def force_directed_init(rng, seed):
    """Force-directed initialization."""
    local_rng = np.random.default_rng(seed)
    c = local_rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(1500):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                dv = c[i] - c[j]
                d = np.linalg.norm(dv)
                if d < 0.25 and d > 1e-4:
                    push = (0.25 - d) * 0.05 / (d + 1e-4)
                    f[i] += dv / d * push
                    f[j] -= dv / d * push
        # Boundary repulsion
        for i in range(N):
            for d_idx in range(2):
                if c[i, d_idx] < 0.1:
                    f[i, d_idx] += 0.05 * (0.1 - c[i, d_idx])
                if c[i, d_idx] > 0.9:
                    f[i, d_idx] -= 0.05 * (c[i, d_idx] - 0.9)
        c += f * 0.01
        c = np.clip(c, 0.03, 0.97)
    return c

def generate_starts(rng):
    """Generate a comprehensive set of starting configurations."""
    starts = []
    
    # Standard hexagonal patterns with various row distributions
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
        [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
        [5, 5, 6, 6, 4], [6, 5, 4, 6, 5], [5, 6, 6, 4, 5],
        [5, 4, 6, 5, 6], [4, 6, 5, 6, 5], [5, 7, 5, 5, 4],
        [7, 5, 6, 5, 3], [5, 6, 6, 5, 4], [6, 6, 6, 4, 4],
        [4, 4, 6, 6, 6], [5, 5, 5, 6, 5], [6, 5, 6, 6, 3],
        [3, 6, 7, 6, 4], [4, 7, 6, 6, 3], [6, 6, 4, 6, 4],
        [5, 6, 6, 6, 3], [6, 4, 6, 6, 4], [4, 6, 6, 5, 5],
        [5, 5, 6, 5, 5], [6, 6, 5, 6, 3], [5, 4, 5, 6, 6]
    ]
    
    for pat in pats:
        if sum(pat) < N:
            continue
        for r0 in [0.088, 0.092, 0.096, 0.100, 0.104, 0.108, 0.112]:
            c = generate_hex_pattern(pat, r0, rng)
            starts.append(c)
    
    # Rotated patterns
    starts.extend(generate_rotated_hex(rng))
    
    # Corner-optimized patterns
    starts.extend(generate_corner_optimized(rng))
    
    # Force-directed initializations
    for s in range(15):
        starts.append(force_directed_init(rng, s))
    
    # Random starts
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        starts.append(c)
    
    # Reflected patterns
    for c_base in starts[:10]:
        # Horizontal reflection
        c_hr = c_base.copy()
        c_hr[:, 0] = 1.0 - c_hr[:, 0]
        c_hr = np.clip(c_hr, 0.05, 0.95)
        starts.append(c_hr)
        # Vertical reflection
        c_vr = c_base.copy()
        c_vr[:, 1] = 1.0 - c_vr[:, 1]
        c_vr = np.clip(c_vr, 0.05, 0.95)
        starts.append(c_vr)
    
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = max(mr, 0.0)
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def micro_perturb_refine(c_best, r_best, s_best, rng):
    """Fine-grained perturbation on individual circles."""
    c_curr = c_best.copy()
    s_curr = s_best
    improved = True
    
    for outer in range(8):
        if not improved:
            break
        improved = False
        for _ in range(200):
            idx = rng.integers(N)
            direction = rng.integers(4)  # 0:x+, 1:x-, 2:y+, 3:y-
            delta = rng.uniform(0.001, 0.008) * (0.85 ** outer)
            
            c_try = c_curr.copy()
            if direction == 0:
                c_try[idx, 0] += delta
            elif direction == 1:
                c_try[idx, 0] -= delta
            elif direction == 2:
                c_try[idx, 1] += delta
            else:
                c_try[idx, 1] -= delta
            
            c_try = np.clip(c_try, 1e-6, 1.0 - 1e-6)
            _, s_try, _ = solve_lp_and_grad(c_try)
            
            if s_try > s_curr + 1e-12:
                c_curr = c_try
                s_curr = s_try
                improved = True
    
    return c_curr, s_curr

def run_packing() -> tuple:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    print(f"Testing {len(starts)} initial configurations...")
    
    # Phase 1: Multi-start optimization with Gradient Ascent + SLSQP
    for i, c_init in enumerate(starts):
        if i % 10 == 0:
            print(f"Processing start {i}/{len(starts)}, best so far: {best_s:.6f}")
        
        # Quick GA first
        c_ga, s_ga = ga_optimize_simple(c_init, max_iter=2500, step_init=0.012)
        
        if s_ga <= best_s - 0.001:
            continue  # Skip if far from best
        
        r_ga, _, _ = solve_lp_and_grad(c_ga)
        
        # SLSQP polish with extended iterations
        c_sl, r_sl, s_sl = slsqp_polish(c_ga, r_ga, max_iter=15000)
        
        curr_s = max(s_ga, s_sl)
        curr_c = c_sl if s_sl > s_ga else c_ga
        curr_r = r_sl if s_sl > s_ga else r_ga
        
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r = curr_r.copy()
            print(f"  New best: {best_s:.6f}")
    
    # Phase 2: Line search gradient ascent on current best
    c_ga2, s_ga2 = ga_optimize_line_search(best_c, max_iter=5000)
    if s_ga2 > best_s:
        best_s = s_ga2
        best_c = c_ga2
        best_r, _, _ = solve_lp_and_grad(best_c)
    
    # Phase 3: Extended perturbation search
    for iter_num in range(80):
        noise_scale = 0.02 * (0.9 ** (iter_num // 10))
        n_pert = rng.integers(3, 12)
        idx = rng.choice(N, size=n_pert, replace=False)
        c_pert = best_c.copy()
        c_pert[idx] += rng.normal(0, noise_scale, (n_pert, 2))
        c_pert = np.clip(c_pert, 0.03, 0.97)
        
        c_kk, s_kk = ga_optimize_simple(c_pert, max_iter=2000, step_init=0.008)
        if s_kk > best_s:
            best_s = s_kk
            best_c = c_kk
            best_r, _, _ = solve_lp_and_grad(best_c)
            print(f"  Perturbation improved: {best_s:.6f}")
            
            # Polish the new best
            c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r, max_iter=10000)
            if s_sl > best_s:
                best_s = s_sl
                best_c = c_sl
                best_r = r_sl
    
    # Phase 4: Basin hopping with simulated annealing
    c_bh = best_c.copy()
    s_bh = best_s
    T = 0.01
    for step in range(1500):
        noise_scale = 0.008 * (0.98 ** (step / 200.0))
        c_try = c_bh + rng.normal(0, noise_scale, c_bh.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_bh
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_bh, s_bh = c_try, s_try
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.995
    
    # Phase 5: Micro perturbation refinement
    best_c, best_s = micro_perturb_refine(best_c, best_r, best_s, rng)
    
    # Phase 6: Final SLSQP polish with maximum effort
    if best_r is None:
        best_r, _, _ = solve_lp_and_grad(best_c)
    c_final, r_final, s_final = slsqp_polish(best_c, best_r, max_iter=25000)
    if s_final > best_s:
        best_c = c_final
        best_r = r_final
        best_s = s_final
    
    # Phase 7: One more line search GA on final result
    c_ga3, s_ga3 = ga_optimize_line_search(best_c, max_iter=3000)
    if s_ga3 > best_s:
        best_s = s_ga3
        best_c = c_ga3
        best_r, _, _ = solve_lp_and_grad(best_c)
    
    # Final LP to ensure radii are optimal for the found centers
    if best_r is None:
        best_r, best_s_final, _ = solve_lp_and_grad(best_c)
        if best_s_final > best_s:
            best_s = best_s_final
    
    # Final strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    print(f"Final sum of radii: {final_sum:.6f}")
    return best_c, radii, final_sum
