# sol_000350 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000344 (state 37d9ed17) state=8c8773eb sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute constant LP constraint matrix structure for efficiency
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
    ub = np.maximum(ub, 1e-12)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
    b = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b[k] = dists[i, j]; k += 1
    for i in range(N):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    bounds = [(0.0, u) for u in ub]
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=bounds, method='highs')
    
    if not res.success:
        return np.zeros(N), 0.0, np.zeros_like(centers), np.zeros(NUM_PAIRS + 4 * N)
        
    radii = res.x
    try:
        duals = np.asarray(res.marginals.ineqlin)
    except AttributeError:
        try:
            duals = np.asarray(res.ineqlin.marginals)
        except AttributeError:
            duals = np.zeros(NUM_PAIRS + 4 * N)
            
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
        
    return radii, np.sum(radii), grad, duals

def lbfgs_func(x):
    """Objective and gradient wrapper for L-BFGS-B minimization."""
    c = x.reshape(N, 2)
    _, val, g, _ = solve_lp_and_grad(c)
    return -val, -g.flatten()

def slsqp_obj(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP: boundaries and squared non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def generate_inits(rng):
    """Generates diverse initial configurations for multi-start optimization."""
    starts = []
    # Hexagonal lattice patterns with varying row counts
    pats = [[6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], [4,6,6,6,4], [5,5,5,5,6], [6,4,6,5,5], [5,4,6,6,5]]
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110, 0.115]:
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
            starts.append(np.clip(np.array(c[:N]), 0.02, 0.98))
            
    # Force-directed repulsion layouts
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(600):
            f = np.zeros_like(c)
            d_mat = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
            d_mat = np.maximum(d_mat, 1e-5)
            inv = 1.0 / d_mat
            np.fill_diagonal(inv, 0.0)
            force = np.where(d_mat < 0.28, inv**2, 0.0)
            diff = c[:, None, :] - c[None, :, :]
            for d in range(2):
                f[:, d] = np.sum(diff[:, :, d] * force / d_mat, axis=1)
            c += f * 0.003
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # Corner-biased starts to encourage boundary utilization
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        c[:4] = corners
        c += rng.normal(0, 0.008, c.shape)
        starts.append(np.clip(c, 0.05, 0.95))
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance within 1e-12 tolerance."""
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
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    rng = np.random.default_rng(42)
    bounds_c = [(0.005, 0.995)] * (2 * N)
    best_c, best_r, best_s = None, None, -1.0
    
    starts = generate_inits(rng)
    
    # Phase 1: Multi-start L-BFGS-B Optimization
    for c0 in starts:
        try:
            res = minimize(lbfgs_func, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_c = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    best_r, _, _, _ = solve_lp_and_grad(best_c)
    
    # Phase 2: Perturbation + L-BFGS-B Refinement
    for _ in range(40):
        c_p = best_c + rng.normal(0, 0.007, best_c.shape)
        c_p = np.clip(c_p, 0.02, 0.98)
        try:
            res = minimize(lbfgs_func, c_p.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c, options={'maxiter': 2500, 'ftol': 1e-13})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_c = res.x.reshape(N, 2).copy()
                best_r, _, _, _ = solve_lp_and_grad(best_c)
        except Exception:
            pass
            
    # Phase 3: Joint SLSQP Polish (centers + radii)
    for _ in range(6):
        v0 = np.concatenate([best_c.flatten(), best_r])
        bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        try:
            res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_sl, constraints={'type': 'ineq', 'fun': slsqp_cons}, options={'maxiter': 8000, 'ftol': 1e-14})
            if np.min(slsqp_cons(res.x)) >= -1e-7:
                s = np.sum(res.x[2 * N:])
                if s > best_s:
                    best_s = s
                    best_c = res.x[:2 * N].reshape(N, 2).copy()
                    best_r = res.x[2 * N:].copy()
        except Exception:
            pass
            
    # Phase 4: Simulated Annealing with Subset Moves
    T = 0.008
    c_curr, s_curr = best_c.copy(), best_s
    for step in range(2000):
        n_move = rng.integers(2, 8)
        idx = rng.choice(N, n_move, replace=False)
        c_try = c_curr.copy()
        c_try[idx] += rng.normal(0, T, (n_move, 2))
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            c_curr, s_curr = c_try, s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r, _, _, _ = solve_lp_and_grad(best_c)
        T *= 0.995
        
    # Phase 5: Shrink-Push-Restart Cycles to escape topological traps
    for cyc in range(5):
        shrink = 0.75 + cyc * 0.03
        # Push centers apart using repulsive forces
        c_push = best_c.copy()
        for _ in range(300):
            f = np.zeros_like(c_push)
            d_mat = np.linalg.norm(c_push[:, None, :] - c_push[None, :, :], axis=2)
            d_mat = np.maximum(d_mat, 1e-5)
            inv = 1.0 / d_mat
            np.fill_diagonal(inv, 0.0)
            force = np.where(d_mat < 0.25, inv**2, 0.0)
            diff = c_push[:, None, :] - c_push[None, :, :]
            for d in range(2):
                f[:, d] = np.sum(diff[:, :, d] * force / d_mat, axis=1)
            c_push += f * 0.002
            c_push = np.clip(c_push, 0.05, 0.95)
            
        try:
            res = minimize(lbfgs_func, c_push.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_c = res.x.reshape(N, 2).copy()
                best_r, _, _, _ = solve_lp_and_grad(best_c)
        except Exception:
            pass
            
    # Final verification and strict repair
    r_final, s_final, _, _ = solve_lp_and_grad(best_c)
    if s_final > best_s:
        best_r = r_final
        best_s = s_final
        
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
