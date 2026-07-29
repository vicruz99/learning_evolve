# sol_000361 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000320 (state 24d66f03) state=b1abfc64 sum of radii=2.613549 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

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

def obj_lp(x):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    c = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def cons_joint(v):
    """Computes boundary and non-overlap constraints for joint SLSQP optimization."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def obj_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def force_init(rng):
    """Generates a well-spaced configuration via repulsive forces."""
    c = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(600):
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(i+1, N):
                dv = c[i] - c[j]
                d = np.linalg.norm(dv)
                if d < 0.2 and d > 1e-4:
                    push = (0.2 - d) * 0.05 / (d + 1e-4)
                    f[i] += dv / d * push
                    f[j] -= dv / d * push
        c += f * 0.5
        c = np.clip(c, 0.05, 0.95)
    return c

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    # Hexagonal lattice patterns
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]]
    for pat in pats:
        for r0 in [0.09, 0.095, 0.10, 0.105]:
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
            
    # Corner and boundary heavy starts
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        c[:4] = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c += rng.normal(0, 0.01, c.shape)
        starts.append(np.clip(c, 0.02, 0.98))
        
    # Force-directed spreads
    for _ in range(15):
        starts.append(force_init(rng))
        
    # Uniform random
    for _ in range(15):
        starts.append(rng.uniform(0.1, 0.9, (N, 2)))
        
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
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def run_packing():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    bounds_c = [(0.001, 0.999)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for c_init in starts:
        try:
            res = minimize(obj_lp, c_init.flatten(), jac=True, method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 5000, 'ftol': 1e-14})
            c1 = res.x.reshape(N, 2)
            r1, s1, _ = solve_lp_and_grad(c1)
            if s1 > best_s:
                best_s = s1
                best_c = c1.copy()
                best_r = r1.copy()
        except Exception:
            pass
            
    # Phase 2: Joint SLSQP Polish
    if best_c is not None:
        v0 = np.concatenate([best_c.flatten(), best_r])
        try:
            res_sl = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_joint,
                              constraints={'type': 'ineq', 'fun': cons_joint},
                              options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(cons_joint(res_sl.x)) >= -1e-8:
                s_sl = np.sum(res_sl.x[2*N:])
                if s_sl > best_s:
                    best_s = s_sl
                    best_c = res_sl.x[:2*N].reshape(N, 2).copy()
                    best_r = res_sl.x[2*N:].copy()
        except Exception:
            pass

    # Phase 3: Optimization-Assisted Simulated Annealing (OASA)
    # Accepts moves based on energy, then immediately runs local optimization to settle
    c_curr = best_c.copy()
    s_curr = best_s
    T = 0.008
    for step in range(2000):
        cluster_size = rng.integers(1, 7)
        idx = rng.choice(N, cluster_size, replace=False)
        c_try = c_curr.copy()
        c_try[idx] += rng.normal(0, T, (cluster_size, 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        
        _, s_try, _ = solve_lp_and_grad(c_try)
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
            c_curr = c_try
            s_curr = s_try
            
            # Quick local optimization after accepting to settle into basin
            try:
                res_loc = minimize(obj_lp, c_curr.flatten(), jac=True, method='L-BFGS-B',
                                   bounds=bounds_c, options={'maxiter': 500, 'ftol': 1e-13})
                c_loc = res_loc.x.reshape(N, 2)
                r_loc, s_loc, _ = solve_lp_and_grad(c_loc)
                if s_loc > s_curr:
                    c_curr = c_loc
                    s_curr = s_loc
            except Exception:
                pass
                
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.998
        
    # Phase 4: Final SLSQP Joint Polish
    v0_fin = np.concatenate([best_c.flatten(), best_r])
    try:
        res_fin = minimize(obj_joint, v0_fin, method='SLSQP', bounds=bounds_joint,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_joint(res_fin.x)) >= -1e-8:
            s_fin = np.sum(res_fin.x[2*N:])
            if s_fin > best_s:
                best_c = res_fin.x[:2*N].reshape(N, 2)
                best_r = res_fin.x[2*N:]
                best_s = s_fin
    except Exception:
        pass
        
    # Phase 5: Strict numerical repair
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
