# sol_000296 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000243 (state e183a9b7) state=b5b9f03a sum of radii=2.297617 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

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
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    # Extract dual marginals safely across scipy versions
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

def lbfgs_optimize(centers0, bounds_xy):
    """Optimize centers using L-BFGS-B with analytic gradient from LP."""
    def obj_and_grad(x_flat):
        c = x_flat.reshape(N, 2)
        c = np.clip(c, 1e-5, 1.0 - 1e-5)
        r, s, g = solve_lp_and_grad(c)
        return -s, -g.flatten()
        
    try:
        res = minimize(obj_and_grad, centers0.flatten(), method='L-BFGS-B', 
                       jac=True, bounds=bounds_xy,
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return centers0, 0.0

def ga_optimize(centers0, max_iter=2000, step_init=0.005):
    """Gradient ascent on centers with adaptive line search."""
    centers = centers0.copy()
    best_c = centers.copy()
    best_s = -1.0
    step = step_init
    
    for _ in range(max_iter):
        r, s, g = solve_lp_and_grad(centers)
        if s > best_s:
            best_s = s
            best_c = centers.copy()
            
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
            
        g_dir = g / gn
        step = min(step, 0.02)
        nc = np.clip(centers + step * g_dir, 1e-5, 1.0 - 1e-5)
        _, ns, _ = solve_lp_and_grad(nc)
        
        if ns > s:
            centers = nc
            step *= 1.1
        else:
            step *= 0.7
            if step < 1e-9:
                break
    return best_c, best_s

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def generate_hex_init(pat, r0, rng):
    """Generates hexagonal lattice initial configuration."""
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
    return np.clip(np.array(c[:N]), 0.05, 0.95)

def run_packing() -> tuple:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    bounds_xy = [(0.0, 1.0)] * (2 * N)
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = []
    
    # Diverse hexagonal patterns
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
        [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
        [5, 7, 5, 5, 4], [6, 6, 6, 4, 4], [5, 5, 7, 5, 4]
    ]
    for pat in pats:
        for r0 in [0.088, 0.095, 0.102, 0.108]:
            starts.append(generate_hex_init(pat, r0, rng))
            
    # Force-directed layouts
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.22 and d > 1e-4:
                        push = (0.22 - d) * 0.04 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Random dense starts
    for _ in range(10):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))

    # Phase 1: Multi-start GA + L-BFGS-B
    for c_init in starts:
        c_ga, s_ga = ga_optimize(c_init, 1500)
        c_lb, s_lb = lbfgs_optimize(c_ga, bounds_xy)
        
        curr_c, curr_s = (c_lb, s_lb) if s_lb > s_ga else (c_ga, s_ga)
        if curr_s > best_s:
            best_s = curr_s
            best_c = curr_c.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 2: Targeted Kicks + Basin Hopping
    c_bh = best_c.copy()
    s_bh = best_s
    T = 0.008
    
    for step in range(800):
        # Perturb a subset of circles
        num_kick = rng.integers(3, 8)
        idx = rng.choice(N, size=num_kick, replace=False)
        c_try = c_bh.copy()
        c_try[idx] += rng.normal(0, T * 0.5, (num_kick, 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        # Local refinement
        c_loc, s_loc = lbfgs_optimize(c_try, bounds_xy)
        
        delta = s_loc - s_bh
        if delta > 0 or (T > 1e-6 and np.exp(delta / T) > rng.random()):
            c_bh, s_bh = c_loc, s_loc
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.994
        
    # Phase 3: Final Polish with L-BFGS-B and GA
    c_pol, s_pol = lbfgs_optimize(best_c, bounds_xy)
    if s_pol > best_s:
        best_c = c_pol
        best_s = s_pol
        best_r, _, _ = solve_lp_and_grad(best_c)
        
    c_ga2, s_ga2 = ga_optimize(best_c, 1000, step_init=0.002)
    if s_ga2 > best_s:
        best_c = c_ga2
        best_s = s_ga2
        best_r, _, _ = solve_lp_and_grad(best_c)
        
    # Phase 4: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
