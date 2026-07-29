# sol_000293 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000243 (state e183a9b7) state=eed132c8 sum of radii=2.613549 correctness=1.0
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
    """Solves LP for maximal radii given centers and computes exact subgradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-12)
    
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
        
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
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
            vec = (centers[i] - centers[j]) / d
            grad[i] += mu * vec
            grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
    return radii, s_sum, grad

def obj_grad(v):
    """Objective and gradient for L-BFGS-B on centers."""
    c = np.clip(v.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def obj_joint(v):
    """Objective for joint SLSQP optimization."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Constraints for joint SLSQP optimization."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = np.concatenate([c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5]]
    for pat in pats:
        for r0 in [0.09, 0.095, 0.10, 0.105]:
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
            c = np.array(c[:N])
            c += rng.normal(0, 0.006, c.shape)
            c = np.clip(c, 0.05, 0.95)
            starts.append(c)
            
    for _ in range(12):
        starts.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = np.array([[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]])
        c[:4] = corners + rng.normal(0, 0.01, (4, 2))
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def run_packing():
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c, best_r, best_s = None, None, -1.0
    starts = generate_starts(rng)
    bounds_xy = [(0.005, 0.995)] * (2 * N)
    
    # Phase 1: Multi-start L-BFGS-B on centers using exact LP gradient
    for c0 in starts:
        try:
            res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', 
                          jac=True, bounds=bounds_xy,
                          options={'maxiter': 600, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Basin Hopping to escape local minima
    c_bh = best_c.copy()
    s_bh = best_s
    T = 0.004
    for step in range(500):
        c_try = c_bh + rng.normal(0, 0.0035, c_bh.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_bh
        if delta > 0 or np.exp(delta / max(T, 1e-9)) > rng.random():
            c_bh, s_bh = c_try, s_try
            if s_bh > best_s:
                best_s = s_bh
                best_c = c_bh.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.996
        
    # Phase 3: SLSQP Joint Polish for precision
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_j = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    for _ in range(3):
        v_pert = v0 + rng.normal(0, 0.0015, v0.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.45)
        
        try:
            res = minimize(obj_joint, v_pert, method='SLSQP', bounds=bounds_j,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            if np.min(cons_joint(res.x)) >= -1e-7:
                s_sl = np.sum(res.x[2*N:])
                if s_sl > best_s:
                    best_s = s_sl
                    best_c = res.x[:2*N].reshape(N, 2)
                    best_r = res.x[2*N:]
                    v0 = res.x.copy()
        except Exception:
            pass

    # Phase 4: Deterministic Repair for strict validation compliance
    radii = best_r.copy()
    centers = best_c.copy()
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
                radii[i] = mr
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
