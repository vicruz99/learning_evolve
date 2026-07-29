# sol_000414 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000370 (state bf8a76d9) state=2e75bcdb sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
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
    """Solves LP for maximal radii given fixed centers and computes exact subgradient via duals."""
    centers = np.clip(centers, 1e-8, 1.0 - 1e-8)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-20)
    
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
        duals = np.maximum(res.marginals.ineqlin, 0.0)
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = np.maximum(res.ineqlin.marginals, 0.0)
        
    grad = np.zeros_like(centers)
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-8:
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

def obj_grad_lbfgsb(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def optimize_lbfgsb(c0, maxiter=15000):
    """Optimizes circle positions using L-BFGS-B with exact gradient."""
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    try:
        res = minimize(obj_grad_lbfgsb, c0.flatten(), method='L-BFGS-B',
                       bounds=bounds, jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared for smoothness)."""
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

def slsqp_polish(c0, r0, maxiter=15000):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False})
        c_val = cons_joint(res.x)
        if np.min(c_val) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def coordinate_descent(c0, rng):
    """Optimizes each circle's position independently to break symmetry and escape local minima."""
    c = c0.copy()
    _, best_s, _ = solve_lp_and_grad(c)
    
    for _ in range(4):
        for i in range(N):
            def obj_single(xy):
                temp = c.copy()
                temp[i] = np.clip(xy, 1e-5, 1.0 - 1e-5)
                _, s, _ = solve_lp_and_grad(temp)
                return -s
                
            try:
                res = minimize(obj_single, c[i], method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12})
                if -res.fun > best_s + 1e-9:
                    c[i] = np.clip(res.x, 1e-5, 1.0 - 1e-5)
                    _, best_s, _ = solve_lp_and_grad(c)
            except Exception:
                pass
    return c

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
            [5, 5, 6, 6, 4], [6, 5, 4, 6, 5], [5, 6, 6, 4, 5],
            [5, 4, 6, 5, 6], [4, 6, 5, 6, 5]]
    
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
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
            starts.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Force-directed layouts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(400):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.045 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Boundary and corner biased starts
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for i in range(N):
            if rng.random() < 0.5:
                c[i, 0] = rng.choice([rng.uniform(0.05, 0.18), rng.uniform(0.82, 0.95)])
            else:
                c[i, 1] = rng.choice([rng.uniform(0.05, 0.18), rng.uniform(0.82, 0.95)])
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to ensure strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-10
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
    """Packs 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B from diverse starts
    for c_init in starts:
        c_opt, s_opt = optimize_lbfgsb(c_init, maxiter=12000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
    best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: SLSQP Polish
    c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r, maxiter=10000)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl
        best_r = r_sl
        
    # Phase 3: Perturbation & Re-optimize loop
    for step in range(200):
        scale = 0.02 * (0.92 ** (step // 12))
        c_k = best_c.copy()
        idx = rng.choice(N, size=N, replace=True)
        c_k[idx] += rng.normal(0, scale, (N, 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        
        # Random swap to break symmetry
        s1, s2 = rng.choice(N, 2, replace=False)
        c_k[[s1, s2]] = c_k[[s2, s1]]
        
        c_opt, s_opt = optimize_lbfgsb(c_k, maxiter=8000)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, best_s, _ = solve_lp_and_grad(best_c)
            c_sl, r_sl, s_sl = slsqp_polish(best_c, best_r, maxiter=8000)
            if s_sl > best_s:
                best_s = s_sl
                best_c = c_sl
                best_r = r_sl

    # Phase 4: Coordinate Descent to escape tight local minima
    best_c = coordinate_descent(best_c, rng)
    _, best_s, _ = solve_lp_and_grad(best_c)
    best_r, _, _ = solve_lp_and_grad(best_c)
    
    # Phase 5: Simulated Annealing
    c_sa = best_c.copy()
    s_sa = best_s
    T = 0.010
    for step in range(2500):
        c_try = c_sa + rng.normal(0, T, c_sa.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        if s_try > s_sa or np.exp((s_try - s_sa) / max(T, 1e-9)) > rng.random():
            c_sa, s_sa = c_try, s_try
            if s_sa > best_s:
                best_s = s_sa
                best_c = c_sa.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.996
        
    # Phase 6: Final High-Precision Alternating Polish
    for _ in range(5):
        best_c, best_s = optimize_lbfgsb(best_c, maxiter=10000)
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        best_c, best_r, best_s = slsqp_polish(best_c, best_r, maxiter=8000)
        
    # Phase 7: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
