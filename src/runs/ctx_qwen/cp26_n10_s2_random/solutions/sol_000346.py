# sol_000346 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000242 (state 71f24e7d) state=1a8fad5c sum of radii=2.626109 correctness=1.0
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
    b_start = NUM_PAIRS + 4 * i
    A_LP[b_start, i] = 1.0
    A_LP[b_start + 1, i] = 1.0
    A_LP[b_start + 2, i] = 1.0
    A_LP[b_start + 3, i] = 1.0

def solve_lp(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and duals."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    idx = 0
    for i, j in PAIR_IDX:
        b[idx] = dists[i, j]
        idx += 1
    for i in range(N):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    bounds = [(0.0, u) for u in ub]
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            try:
                duals = res.marginals.ineqlin
            except AttributeError:
                duals = np.zeros(len(b))
            return res.x, np.sum(res.x), duals
    except Exception:
        pass
    return np.zeros(N), 0.0, np.zeros(len(b))

def compute_grad(centers, duals):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    grad = np.zeros_like(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
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
    return grad

def obj_grad_lbfgs(x_flat):
    """Objective and gradient wrapper for L-BFGS-B."""
    c = np.clip(x_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, s, d = solve_lp(c)
    g = compute_grad(c, d)
    return -s, -g.flatten()

def optimize_centers_lbfgs(c0):
    """Optimizes circle positions using L-BFGS-B with exact gradient."""
    bounds = [(1e-5, 0.999)] * (2 * N)
    try:
        res = minimize(obj_grad_lbfgs, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': 6000, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def slsqp_joint_polish(c0, r0):
    """Joint SLSQP optimization of centers and radii for final precision."""
    v0 = np.concatenate([c0.flatten(), r0])
    
    def obj(v):
        return -np.sum(v[2 * N:])
        
    def cons(v):
        c = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
        dx = c[TRIU_I, 0] - c[TRIU_J, 0]
        dy = c[TRIU_I, 1] - c[TRIU_J, 1]
        dr = r[TRIU_I] + r[TRIU_J]
        con.append(dx**2 + dy**2 - dr**2)
        return np.concatenate(con)
        
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons(res.x)) >= -1e-7:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, 0.0

def simulated_annealing_centers(centers, rng):
    """Simulated annealing to escape local optima on center positions."""
    c_curr = centers.copy()
    _, best_s, _ = solve_lp(c_curr)
    best_c = c_curr.copy()
    s_curr = best_s
    
    T = 0.005
    for step in range(3000):
        i = rng.integers(N)
        c_try = c_curr.copy()
        c_try[i] += rng.normal(0, 0.008, 2)
        c_try = np.clip(c_try, 0.01, 0.99)
        
        _, s_try, _ = solve_lp(c_try)
        delta = s_try - s_curr
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_curr = c_try
            s_curr = s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
        T *= 0.9945
        
    return best_c

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
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
    
    # Phase 1: Generate diverse initial configurations
    starts = []
    
    # 1. Hexagonal lattice patterns (known to be near-optimal for N=26)
    patterns = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], [4, 6, 5, 6, 5]]
    for pat in patterns:
        for r_est in [0.098, 0.102, 0.105]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    if len(c) < N:
                        c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
            c = np.array(c[:N])
            # Center and scale tightly
            c = (c - c.min(axis=0)) / (c.max(axis=0) - c.min(axis=0)) * 0.92 + 0.04
            c += rng.normal(0, 0.002, c.shape)
            starts.append(np.clip(c, 0.02, 0.98))
            
    # 2. Force-directed repulsion layouts
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(1000):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d < 0.22 and d > 1e-4:
                        push = (0.22 - d) * 0.05
                        f[i] += d_vec / d * push
                        f[j] -= d_vec / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 3. Corner-biased starts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        starts.append(np.clip(c, 0.05, 0.95))

    # Phase 2: Multi-start L-BFGS-B optimization
    for c_init in starts:
        c_opt, s_opt = optimize_centers_lbfgs(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        _, best_s, _ = solve_lp(best_c)
        
    # Phase 3: Simulated Annealing to escape local minima
    best_c = simulated_annealing_centers(best_c, rng)
    _, best_s, _ = solve_lp(best_c)
    
    # Re-optimize SA result with L-BFGS-B to settle
    c_opt, s_opt = optimize_centers_lbfgs(best_c)
    if s_opt > best_s:
        best_s = s_opt
        best_c = c_opt.copy()
        
    # Phase 4: Targeted perturbations (kick & recover)
    for _ in range(20):
        c_k = best_c.copy()
        idx = rng.choice(N, size=6, replace=False)
        c_k[idx] += rng.normal(0, 0.015, (6, 2))
        c_k = np.clip(c_k, 0.05, 0.95)
        c_kk, s_kk = optimize_centers_lbfgs(c_k)
        if s_kk > best_s:
            best_s = s_kk
            best_c = c_kk.copy()
            
    # Phase 5: Joint SLSQP Polish
    r_lp, _, _ = solve_lp(best_c)
    c_sl, r_sl, s_sl = slsqp_joint_polish(best_c, r_lp)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl.copy()
        best_r = r_sl.copy()
    else:
        best_r = r_lp.copy()
        
    # Final LP solve to ensure radii match centers exactly
    r_final, s_final, _ = solve_lp(best_c)
    if s_final > best_s:
        best_s = s_final
        best_r = r_final.copy()
        
    # Phase 6: Strict numerical repair
    radii = repair(best_c, best_r)
    
    return best_c, radii, float(np.sum(radii))
