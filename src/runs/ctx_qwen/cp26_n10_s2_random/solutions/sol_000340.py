# sol_000340 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=adc172df sum of radii=2.323418 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure
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
    """Solves LP for maximal radii given fixed centers and computes exact gradient via duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
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
        
    bounds = [(0.0, u) for u in ub]
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    except Exception:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        try:
            duals = np.asarray(res.marginals.ineqlin)
        except Exception:
            pass
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except Exception:
            pass
            
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

def lbfgs_wrapper(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x_flat.reshape(N, 2), 1e-5, 1.0 - 1e-5)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def run_optimization(c0, max_iter=8000):
    """Runs L-BFGS-B optimization from initial centers c0."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(lbfgs_wrapper, c0.flatten(), method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        c_opt = np.clip(res.x.reshape(N, 2), 1e-5, 1.0 - 1e-5)
        _, s_opt, _ = solve_lp_and_grad(c_opt)
        return c_opt, s_opt
    except Exception:
        return c0, 0.0

def generate_starts(rng):
    """Generates a diverse set of initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice patterns with varying densities
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5],
            [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5]]
    for pat in pats:
        for r0 in [0.088, 0.094, 0.100, 0.106]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.002), y + rng.normal(0, 0.002)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # 2. Force-directed repulsion layouts
    for _ in range(12):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.24 and d > 1e-4:
                        push = (0.24 - d) * 0.06 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 3. Corner and edge biased starts
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[4:8] = [[0.08, 0.5], [0.92, 0.5], [0.5, 0.08], [0.5, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))
        
    # 4. Pure random starts
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(250):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Clamp to boundaries
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_s = -1.0
    best_r = None
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c_init in starts:
        c_opt, s_opt = run_optimization(c_init, max_iter=10000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        _, best_s, _ = solve_lp_and_grad(best_c)
        
    r_best, _, _ = solve_lp_and_grad(best_c)
    best_r = r_best
    
    # Phase 2: Simulated Annealing / Basin Hopping to escape local minima
    c_curr = best_c.copy()
    s_curr = best_s
    
    for step in range(200):
        # Decaying perturbation scale
        scale = 0.018 * (0.96 ** (step // 15))
        c_pert = c_curr + rng.normal(0, scale, c_curr.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        
        c_opt, s_opt = run_optimization(c_pert, max_iter=5000)
        
        # Metropolis criterion
        delta = s_opt - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(0.0005, (200 - step) * 0.00005)):
            c_curr = c_opt.copy()
            s_curr = s_opt
            
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
    # Phase 3: Final LP polish on best centers found
    r_final, s_final, _ = solve_lp_and_grad(best_c)
    best_r = r_final
    best_s = s_final
    
    # Phase 4: Strict numerical repair
    radii = repair(best_c, best_r)
    
    return best_c, radii, float(np.sum(radii))
