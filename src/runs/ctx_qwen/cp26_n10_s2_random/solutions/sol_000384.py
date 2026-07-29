# sol_000384 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000369 (state f32845d7) state=b76065e2 sum of radii=2.610168 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure for speed
A_LP = np.zeros((NUM_PAIRS + 4 * N, N))
PAIR_IDX = []
lp_idx = 0
for i in range(N):
    for j in range(i + 1, N):
        A_LP[lp_idx, i] = 1.0
        A_LP[lp_idx, j] = 1.0
        PAIR_IDX.append((i, j))
        lp_idx += 1
for i in range(N):
    base = NUM_PAIRS + 4 * i
    A_LP[base, i] = 1.0
    A_LP[base + 1, i] = 1.0
    A_LP[base + 2, i] = 1.0
    A_LP[base + 3, i] = 1.0

def solve_lp_and_grad(centers):
    """Solves LP for maximal radii given fixed centers and computes exact subgradient via duals."""
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
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, u) for u in ub], method='highs')
        if not res.success:
            return np.full(N, 0.05), 0.0, np.zeros_like(centers)
    except Exception:
        return np.full(N, 0.05), 0.0, np.zeros_like(centers)
        
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

def lp_wrapper(x_flat):
    """Wrapper for scipy minimization: returns objective and gradient."""
    c = np.clip(x_flat.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def run_lbfgsb(c0):
    """Optimizes circle positions using L-BFGS-B with exact LP gradient."""
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    try:
        res = minimize(lp_wrapper, c0.flatten(), method='L-BFGS-B',
                       bounds=bounds, jac=True,
                       options={'maxiter': 60000, 'ftol': 1e-16, 'gtol': 1e-14})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
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

def run_slsqp(c0, r0):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 30000, 'ftol': 1e-16, 'disp': False})
        c_val = slsqp_cons(res.x)
        if np.min(c_val) >= -1e-8:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # Hexagonal patterns
    pats = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
        [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5],
        [5, 5, 6, 6, 4], [6, 5, 4, 6, 5], [5, 6, 6, 4, 5],
        [5, 4, 6, 5, 6], [4, 6, 5, 6, 5], [6, 6, 4, 5, 5],
        [5, 7, 5, 5, 4], [4, 5, 7, 5, 5], [6, 5, 5, 5, 5],
        [5, 5, 6, 5, 5], [5, 6, 5, 5, 5]
    ]
    
    for pat in pats:
        for r0 in [0.088, 0.094, 0.100, 0.106, 0.112]:
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
            starts.append(np.clip(np.array(c[:N]), 0.02, 0.98))
            
    # Boundary & Corner biased
    for _ in range(20):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        edges = [[0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        fixed = corners + edges
        c[:len(fixed)] = fixed
        c += rng.normal(0, 0.02, c.shape)
        starts.append(np.clip(c, 0.02, 0.98))
        
    # Force-directed relaxation
    for _ in range(15):
        c = rng.uniform(0.1, 0.9, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            diffs = c[:, None, :] - c[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            dists = np.maximum(dists, 1e-4)
            rep = np.where(dists < 0.25, 0.02 / (dists**2 + 1e-4), 0.0)
            f = np.sum(diffs * rep[:, :, None], axis=1)
            c += 0.005 * f
            c = np.clip(c, 0.02, 0.98)
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to ensure strict validation compliance with minimal shrinkage."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: L-BFGS-B from diverse starts
    for c_init in starts:
        c_opt, s_opt = run_lbfgsb(c_init)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_s, _ = solve_lp_and_grad(best_c)
    else:
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: SLSQP Polish
    c_sl, r_sl, s_sl = run_slsqp(best_c, best_r)
    if s_sl > best_s:
        best_s = s_sl
        best_c = c_sl
        best_r = r_sl
        
    # Phase 3: Topology Swaps & Re-optimize
    for step in range(60):
        # Random pair swap
        idx_swap = rng.choice(N, 2, replace=False)
        c_swap = best_c.copy()
        c_swap[idx_swap] = c_swap[idx_swap[::-1]]
        
        c_opt, s_opt = run_lbfgsb(c_swap)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
            c_sl, r_sl, s_sl = run_slsqp(best_c, best_r)
            if s_sl > best_s:
                best_s = s_sl
                best_c = c_sl
                best_r = r_sl
                
    # Phase 4: Perturbation & Re-optimize loop
    for step in range(80):
        scale = 0.018 * (0.86 ** (step // 10))
        c_k = best_c.copy()
        idx = rng.choice(N, size=N, replace=True)
        c_k[idx] += rng.normal(0, scale, (N, 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        
        c_opt, s_opt = run_lbfgsb(c_k)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r, _, _ = solve_lp_and_grad(best_c)
            
            c_sl, r_sl, s_sl = run_slsqp(best_c, best_r)
            if s_sl > best_s:
                best_s = s_sl
                best_c = c_sl
                best_r = r_sl
                
    # Phase 5: Simulated Annealing with cluster moves
    c_sa = best_c.copy()
    s_sa = best_s
    T = 0.015
    for step in range(3000):
        n_move = rng.integers(2, 10)
        idx = rng.choice(N, n_move, replace=False)
        c_try = c_sa.copy()
        c_try[idx] += rng.normal(0, T, (n_move, 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_sa
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_sa, s_sa = c_try, s_try
            if s_sa > best_s:
                best_s = s_sa
                best_c = c_sa.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.994
        
        # Occasional local restart during SA
        if step % 200 == 0 and step > 0:
            c_opt, s_opt = run_lbfgsb(c_sa)
            if s_opt > s_sa:
                c_sa, s_sa = c_opt, s_opt
                if s_sa > best_s:
                    best_s = s_sa
                    best_c = c_sa.copy()
                    best_r, _, _ = solve_lp_and_grad(best_c)
                    
    # Phase 6: Final Joint Polish & Repair
    c_final, r_final, s_final = run_slsqp(best_c, best_r)
    if s_final > best_s:
        best_c = c_final
        best_r = r_final
        best_s = s_final
        
    # One last precise L-BFGS-B polish
    c_precise, s_precise = run_lbfgsb(best_c)
    if s_precise > best_s:
        best_c = c_precise
        best_s = s_precise
        best_r, _, _ = solve_lp_and_grad(best_c)
        
    # Strict numerical repair
    radii = repair(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
