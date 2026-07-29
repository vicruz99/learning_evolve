# sol_000239 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000213 (state adb87445) state=1f5985db sum of radii=2.624554 correctness=1.0
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

def solve_lp(centers):
    """Solves LP for maximal radii given fixed centers. Returns radii, sum, and dual multipliers."""
    n = centers.shape[0]
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
    for i in range(n):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, 
                  bounds=[(0.0, u) for u in ub], method='highs')
    if res.success:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except AttributeError:
            duals = np.zeros(len(b))
        return res.x, np.sum(res.x), duals
    return np.zeros(n), 0.0, np.zeros(len(b))

def objective_and_grad(centers_flat):
    """Objective (negative sum of radii) and exact gradient w.r.t flattened centers."""
    centers = centers_flat.reshape(N, 2)
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    radii, s, duals = solve_lp(centers)
    
    grad_flat = np.zeros_like(centers_flat)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    k = 0
    for i, j in PAIR_IDX:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad_flat[2*i] += mu * vec[0]
                grad_flat[2*i+1] += mu * vec[1]
                grad_flat[2*j] -= mu * vec[0]
                grad_flat[2*j+1] -= mu * vec[1]
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad_flat[2*i] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad_flat[2*i+1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return -s, grad_flat

def optimize_centers_lbfgs(c0):
    """Optimizes centers using L-BFGS-B with exact LP gradient."""
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    try:
        res = minimize(objective_and_grad, c0.flatten(), method='L-BFGS-B',
                       bounds=bounds, jac=True,
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def slsqp_joint(c0, r0):
    """Joint SLSQP optimization of centers and radii using smooth squared constraints."""
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

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c, best_r, best_s = None, None, -1.0
    
    # Generate diverse initial configurations
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]]
    for pat in pats:
        for r0 in [0.092, 0.098, 0.105]:
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
            
    for _ in range(15):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Multi-start optimization loop
    for c_init in starts:
        # Phase 1: Gradient-based center optimization
        c1, s1 = optimize_centers_lbfgs(c_init)
        r1, _, _ = solve_lp(c1)
        
        # Phase 2: Joint SLSQP polish
        c2, r2, s2 = slsqp_joint(c1, r1)
        
        c_curr, r_curr, s_curr = (c2, r2, s2) if s2 > s1 else (c1, r1, s1)
        
        # Phase 3: Basin Hopping / Simulated Annealing on centers
        c_bh, s_bh, r_bh = c_curr, s_curr, r_curr
        T = 0.006
        for step in range(120):
            c_try = c_bh + rng.normal(0, 0.006, c_bh.shape)
            c_try = np.clip(c_try, 0.02, 0.98)
            _, s_try, _ = solve_lp(c_try)
            
            if s_try > s_bh or np.exp((s_try - s_bh) / T) > rng.random():
                c_bh, s_bh = c_try, s_try
                if s_bh > best_s:
                    best_s = s_bh
                    best_c = c_bh.copy()
                    best_r, _, _ = solve_lp(best_c)
            T *= 0.985
            
        if s_bh > best_s:
            best_s = s_bh
            best_c = c_bh.copy()
            best_r, _, _ = solve_lp(best_c)
            
    # Final LP solve to match radii exactly to best centers
    r_final, s_final, _ = solve_lp(best_c)
    if s_final > best_s:
        best_s = s_final
        best_r = r_final
        
    # Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
