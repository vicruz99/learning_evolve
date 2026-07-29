# sol_000404 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000353 (state 4ca32851) state=d44af0a0 sum of radii=2.610158 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2
NUM_CONSTRAINTS = NUM_PAIRS + 4 * N

# Precompute LP constraint matrix structure globally
A_LP = np.zeros((NUM_CONSTRAINTS, N))
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
    """Solves LP for maximal radii given fixed centers and computes exact subgradient."""
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b_ub = np.zeros(NUM_CONSTRAINTS)
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    bounds = [(0.0, u) for u in ub]
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return np.full(N, 0.05), 1.3, np.zeros_like(centers)
    except Exception:
        return np.full(N, 0.05), 1.3, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(NUM_CONSTRAINTS)
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
        
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

def lbfgs_obj(x_flat):
    """Objective for L-BFGS-B: minimizes negative sum of radii."""
    c = x_flat.reshape(N, 2)
    _, s, _ = solve_lp_and_grad(c)
    return -s

def lbfgs_grad(x_flat):
    """Gradient for L-BFGS-B."""
    c = x_flat.reshape(N, 2)
    _, _, g = solve_lp_and_grad(c)
    return -g.flatten()

def optimize_centers_lbfgs(c0, max_iter=3000):
    """Optimizes centers using L-BFGS-B with exact LP gradients."""
    bounds = [(0.005, 0.995)] * (2 * N)
    try:
        res = minimize(lbfgs_obj, c0.flatten(), jac=lbfgs_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-11})
        return res.x.reshape(N, 2), -res.fun
    except Exception:
        return c0, 0.0

def coordinate_optimize(c0, rng):
    """Refines each circle's position independently to improve local contacts."""
    c = c0.copy()
    _, best_s, _ = solve_lp_and_grad(c)
    
    for _ in range(3):
        for i in range(N):
            best_pos = c[i].copy()
            best_val = best_s
            for _ in range(8):
                d = rng.normal(0, 1, 2)
                d /= np.linalg.norm(d)
                for step in [0.005, -0.005, 0.01, -0.01]:
                    c_try = c.copy()
                    c_try[i] = np.clip(c[i] + d * step, 0.01, 0.99)
                    _, s_val, _ = solve_lp_and_grad(c_try)
                    if s_val > best_val + 1e-9:
                        best_val = s_val
                        best_pos = c_try[i].copy()
            c[i] = best_pos
            _, best_s, _ = solve_lp_and_grad(c)
            
    return c, best_s

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundary and non-overlap (squared)."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = np.concatenate([c[:, 0] - r, 1.0 - c[:, 0] - r, 
                          c[:, 1] - r, 1.0 - c[:, 1] - r])
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con = np.concatenate([con, dx**2 + dy**2 - dr**2])
    return con

def slsqp_joint_polish(centers, radii):
    """Joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
        if np.min(slsqp_cons(res.x)) >= -1e-8:
            return res.x[:2 * N].reshape(N, 2), res.x[2 * N:], -res.fun
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(150):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
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

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
            [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5], [5, 5, 5, 5, 6]]
    
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
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
            c += rng.normal(0, 0.002, c.shape)
            starts.append(np.clip(c, 0.05, 0.95))
            
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        f[i] += dv / d * (0.25 - d) * 0.04 / (d + 1e-4)
                        f[j] -= dv / d * (0.25 - d) * 0.04 / (d + 1e-4)
            c += f * 0.005
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    for _ in range(10):
        c = rng.uniform(0.1, 0.9, (N, 2))
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
        c[:4] = corners
        starts.append(np.clip(c, 0.02, 0.98))
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B
    for c0 in starts:
        c_opt, s_opt = optimize_centers_lbfgs(c0, 3500)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            
    if best_c is not None:
        best_r, best_s, _ = solve_lp_and_grad(best_c)
        
        # Phase 2: Coordinate-wise Refinement
        best_c, best_s = coordinate_optimize(best_c, rng)
        best_r, _, _ = solve_lp_and_grad(best_c)
        
        # Phase 3: Simulated Annealing with L-BFGS-B refinement
        c_curr = best_c.copy()
        s_curr = best_s
        T = 0.009
        
        for step in range(1200):
            strategy = rng.integers(0, 3)
            c_try = c_curr.copy()
            scale = 0.016 * (1.0 - step / 1300.0)
            
            if strategy == 0:
                idx = rng.choice(N, size=rng.integers(2, 6), replace=False)
                c_try[idx] += rng.normal(0, scale, (len(idx), 2))
            elif strategy == 1:
                i, j = rng.choice(N, 2, replace=False)
                c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
            else:
                c_try += rng.normal(0, scale * 0.5, c_try.shape)
                
            c_try = np.clip(c_try, 0.02, 0.98)
            
            c_ref, s_ref = optimize_centers_lbfgs(c_try, 2000)
            
            delta = s_ref - s_curr
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                c_curr = c_ref
                s_curr = s_ref
                if s_curr > best_s:
                    best_s = s_curr
                    best_c = c_curr.copy()
                    best_r, _, _ = solve_lp_and_grad(best_c)
            T *= 0.996
            
        # Phase 4: Swap-based Topological Search
        for _ in range(60):
            i, j = rng.choice(N, 2, replace=False)
            c_swap = best_c.copy()
            c_swap[i], c_swap[j] = c_swap[j].copy(), c_swap[i].copy()
            c_sw, s_sw = optimize_centers_lbfgs(c_swap, 2500)
            if s_sw > best_s:
                best_s = s_sw
                best_c = c_sw
                best_r, _, _ = solve_lp_and_grad(best_c)
                
        # Phase 5: Final SLSQP Joint Polish
        c_final, r_final, s_final = slsqp_joint_polish(best_c, best_r)
        if s_final > best_s:
            best_s = s_final
            best_c = c_final
            best_r = r_final
            
    # Phase 6: Strict numerical repair
    radii = repair(best_c, best_r)
    final_sum = float(np.sum(radii))
    
    return best_c, radii, final_sum
