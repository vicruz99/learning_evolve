# sol_000345 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=dd8feb7b sum of radii=2.611776 correctness=1.0
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
k = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[k, i] = 1.0
        A_LP[k, j] = 1.0
        PAIR_IDX.append((i, j))
        k += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and gradient."""
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
        
    bounds_r = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds_r, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b))
    if hasattr(res, 'marginals') and res.marginals is not None:
        try:
            duals = res.marginals.ineqlin
        except Exception:
            pass
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        try:
            duals = res.ineqlin.marginals
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

def obj_grad_func(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    centers = x_flat.reshape(N, 2)
    centers = np.clip(centers, 0.001, 0.999)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g.flatten()

def local_obj_2d(xy, centers, i):
    """Objective for coordinate-wise refinement."""
    temp = centers.copy()
    temp[i] = np.clip(xy, 0.001, 0.999)
    _, s, _ = solve_lp_and_grad(temp)
    return -s

def joint_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def joint_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    return np.concatenate([con, dx**2 + dy**2 - dr**2])

def optimize_lbfgsb(centers0):
    """Optimizes circle positions using L-BFGS-B with exact gradient."""
    bounds = [(0.001, 0.999)] * (2 * N)
    try:
        res = minimize(obj_grad_func, centers0.flatten(), jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 4000, 'ftol': 1e-14})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return centers0, 0.0

def coordinate_refine(centers0):
    """Refines each circle's position independently using Nelder-Mead."""
    centers = centers0.copy()
    _, best_s, _ = solve_lp_and_grad(centers)
    for _ in range(2):
        for i in range(N):
            try:
                res = minimize(local_obj_2d, centers[i], args=(centers, i), method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-12})
                if -res.fun > best_s + 1e-9:
                    centers[i] = np.clip(res.x, 0.001, 0.999)
                    _, best_s, _ = solve_lp_and_grad(centers)
            except Exception:
                pass
    return centers, best_s

def slsqp_joint(c0, r0):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(joint_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': joint_cons},
                       options={'maxiter': 5000, 'ftol': 1e-14})
        if np.min(joint_cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    # Hexagonal lattice patterns
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4]]
    for pat in pats:
        for r0 in [0.095, 0.100, 0.105]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 == 1 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Force-directed repulsion layouts
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.05
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Corner-biased starts to exploit boundary space
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))
        
    # Random starts
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return starts

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
        c_opt, s_opt = optimize_lbfgsb(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        _, best_s, _ = solve_lp_and_grad(best_c)
        
    best_r, best_s, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Coordinate-wise refinement to escape shallow local minima
    best_c, best_s = coordinate_refine(best_c)
    best_r, best_s, _ = solve_lp_and_grad(best_c)
    
    # Phase 3: Simulated Annealing to explore global configuration space
    curr_c = best_c.copy()
    curr_s = best_s
    T = 0.008
    for step in range(600):
        c_try = curr_c.copy()
        idx = rng.choice(N, size=max(2, N // 5), replace=False)
        scale = 0.006 * (0.998 ** step)
        c_try[idx] += rng.normal(0, scale, c_try[idx].shape)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - curr_s
        if delta > 0 or rng.random() < np.exp(min(0.0, delta / max(T, 1e-9))):
            curr_c = c_try
            curr_s = s_try
            if curr_s > best_s:
                best_s = curr_s
                best_c = curr_c.copy()
                best_r, best_s, _ = solve_lp_and_grad(best_c)
        T *= 0.996
        
    # Refine best SA result with L-BFGS-B
    c_opt, s_opt = optimize_lbfgsb(best_c)
    if s_opt > best_s:
        best_s = s_opt
        best_c = c_opt
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 4: Joint SLSQP polish for final precision
    c_sl, r_sl, s_sl = slsqp_joint(best_c, best_r)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl
        best_r = r_sl
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
