# sol_000342 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=b6614912 sum of radii=2.400483 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure globally
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
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-15)
    
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
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    
    # Extract duals robustly across scipy versions
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        try:
            duals = np.asarray(res.marginals.ineqlin).flatten()
        except Exception:
            pass
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        try:
            duals = np.asarray(res.ineqlin.marginals).flatten()
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
        
    return radii, np.sum(radii), grad

def obj_grad_lbfgsb(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
    _, val, grad = solve_lp_and_grad(centers)
    return -val, -grad.flatten()

def optimize_lbgbs(c0):
    """Optimizes circle positions using L-BFGS-B with exact LP gradient."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(obj_grad_lbfgsb, c0.flatten(), method='L-BFGS-B',
                       bounds=bounds, jac=True,
                       options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def force_directed_init(rng):
    """Generates a well-spaced configuration via repulsive forces."""
    c = rng.uniform(0.2, 0.8, (N, 2))
    for _ in range(800):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = c[i] - c[j]
                d = np.linalg.norm(d_vec)
                if d < 0.22 and d > 1e-4:
                    push = (0.22 - d) * 0.06 / (d + 1e-4)
                    f[i] += d_vec / d * push
                    f[j] -= d_vec / d * push
        # Gentle boundary repulsion
        f += np.where(c < 0.1, 0.08 * (0.1 - c), 0.0)
        f -= np.where(c > 0.9, 0.08 * (c - 0.9), 0.0)
        c += f * 0.008
        c = np.clip(c, 0.05, 0.95)
    return c

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5],
            [5, 5, 5, 5, 6], [5, 6, 6, 4, 5], [4, 5, 6, 5, 6],
            [6, 5, 5, 5, 5], [5, 4, 6, 5, 6]]
            
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x + rng.normal(0, 0.004), y + rng.normal(0, 0.004)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    for _ in range(15):
        starts.append(force_directed_init(rng))
        
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        edges = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
        c[:4] = corners
        c[4:8] = edges
        starts.append(np.clip(c, 0.05, 0.95))
        
    return starts

def simulated_annealing(centers, rng, T0=0.006, decay=0.996, steps=1500):
    """Simulated annealing to escape local optima."""
    c_curr = centers.copy()
    _, s_curr, _ = solve_lp_and_grad(c_curr)
    best_c = c_curr.copy()
    best_s = s_curr
    T = T0
    for _ in range(steps):
        i = rng.integers(N)
        c_try = c_curr.copy()
        c_try[i] += rng.normal(0, 0.005, 2)
        c_try = np.clip(c_try, 0.01, 0.99)
        _, s_try, _ = solve_lp_and_grad(c_try)
        if s_try is None: 
            continue
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_curr = c_try
            s_curr = s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
        T *= decay
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
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
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
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c_init in starts:
        c_opt, s_opt = optimize_lbgbs(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
    else:
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Perturbation search to escape local minima
    for step in range(50):
        scale = 0.012 * (0.88 ** (step // 8))
        c_k = best_c.copy()
        idx = rng.choice(N, size=N, replace=True)
        c_k[idx] += rng.normal(0, scale, (N, 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        c_opt, s_opt = optimize_lbgbs(c_k)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, best_s, _ = solve_lp_and_grad(best_c)
            
    # Phase 3: Simulated Annealing for fine-tuning
    c_sa, s_sa = simulated_annealing(best_c, rng)
    if s_sa > best_s:
        best_s = s_sa
        best_c = c_sa
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 4: Final L-BFGS-B polish
    c_final, s_final = optimize_lbgbs(best_c)
    if s_final > best_s:
        best_s = s_final
        best_c = c_final
        
    # Ensure radii exactly match the optimal LP solution for the final centers
    r_final, _, _ = solve_lp_and_grad(best_c)
    best_r = r_final
    
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
