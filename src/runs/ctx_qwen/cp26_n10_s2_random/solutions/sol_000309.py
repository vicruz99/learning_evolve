# sol_000309 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000287 (state 4c08251c) state=2e422364 sum of radii=2.590322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
NUM_PAIRS = N * (N - 1) // 2
NUM_BOUND = 4 * N
TRIU_I, TRIU_J = np.triu_indices(N, 1)

# Precompute LP constraint matrix structure globally
A_LP = np.zeros((NUM_PAIRS + NUM_BOUND, N))
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
    """Solves LP for maximal radii given fixed centers and computes exact subgradient via duals."""
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    b_ub = np.zeros(NUM_PAIRS + NUM_BOUND)
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(n):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.ones(n) * 0.05, 0.0, np.zeros_like(centers)
    except Exception:
        return np.ones(n) * 0.05, 0.0, np.zeros_like(centers)
        
    radii = res.x
    s_sum = np.sum(radii)
    
    duals = np.zeros(len(b_ub))
    if hasattr(res, 'marginals') and res.marginals is not None:
        duals = res.marginals.ineqlin
    elif hasattr(res, 'ineqlin') and res.ineqlin is not None:
        duals = res.ineqlin.marginals
        
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
    for i in range(n):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def obj_grad_lbfgs(v):
    """Objective and exact gradient for L-BFGS-B optimization of centers."""
    c = np.clip(v.reshape(N, 2), 1e-6, 1.0 - 1e-6)
    _, val, grad = solve_lp_and_grad(c)
    return -val, -grad.flatten()

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal patterns with varying row distributions and radii
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 5, 5, 6],
            [5, 4, 6, 6, 5], [4, 5, 6, 5, 6]]
    for pat in pats:
        for r0 in [0.095, 0.100, 0.105, 0.110]:
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
            
    # 2. Force-directed layouts (promote even spacing)
    for _ in range(15):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(800):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i + 1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.20 and d > 1e-4:
                        push = (0.20 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f * 0.01
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    # 3. Corner/Edge biased starts (exploits boundary space)
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        c[:4] = corners
        c += rng.normal(0, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    return starts

def repair(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr + 1e-12:
                radii[i] = mr
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def obj_sl(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_sl(v):
    """Constraints for SLSQP: boundaries and non-overlap using squared distances."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    dx = c[TRIU_I, 0] - c[TRIU_J, 0]
    dy = c[TRIU_I, 1] - c[TRIU_J, 1]
    dr = r[TRIU_I] + r[TRIU_J]
    con.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.01, 0.99)] * (2 * N)
    
    best_c = None
    best_sum = -1.0
    best_r = None
    
    starts = generate_starts(rng)
    
    # Phase 1: Multi-start L-BFGS-B Center Optimization
    for c0 in starts:
        try:
            res = minimize(obj_grad_lbfgs, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = starts[0]
        best_r, best_sum, _ = solve_lp_and_grad(best_c)
        
    # Phase 2: Simulated Annealing with periodic local refinement
    curr_c = best_c.copy()
    curr_sum = best_sum
    T = 0.006
    
    for step in range(600):
        scale = 0.009 * (0.996**step)
        c_trial = curr_c + rng.normal(0, scale, curr_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        r_trial, s_trial, _ = solve_lp_and_grad(c_trial)
        
        if s_trial > best_sum:
            best_sum = s_trial
            best_c = c_trial.copy()
            best_r = r_trial.copy()
            
        delta = s_trial - curr_sum
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
            curr_c = c_trial
            curr_sum = s_trial
            
        T *= 0.997
        
        # Periodic local gradient descent to settle
        if step % 25 == 0 and step > 0:
            try:
                res = minimize(obj_grad_lbfgs, curr_c.flatten(), method='L-BFGS-B', jac=True,
                               bounds=bounds_lbfgs, options={'maxiter': 1200, 'ftol': 1e-13})
                c_loc = res.x.reshape(N, 2)
                r_loc, s_loc, _ = solve_lp_and_grad(c_loc)
                if s_loc > curr_sum:
                    curr_c = c_loc
                    curr_sum = s_loc
                    if s_loc > best_sum:
                        best_sum = s_loc
                        best_c = c_loc.copy()
                        best_r = r_loc.copy()
            except Exception:
                pass
                
    # Phase 3: Decayed perturbation search to escape deep local minima
    for step in range(40):
        scale = 0.016 * (0.86 ** step)
        c_trial = best_c + rng.normal(0, scale, best_c.shape)
        c_trial = np.clip(c_trial, 0.02, 0.98)
        
        try:
            res = minimize(obj_grad_lbfgs, c_trial.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_lbfgs, options={'maxiter': 2500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_sum:
                best_sum = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 4: SLSQP Joint Polish for final precision
    v0 = np.concatenate([best_c.flatten(), best_r])
    bounds_sl = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(obj_sl, v0, method='SLSQP', bounds=bounds_sl,
                       constraints={'type': 'ineq', 'fun': cons_sl},
                       options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
        if np.min(cons_sl(res.x)) >= -1e-8:
            s = np.sum(res.x[2 * N:])
            if s > best_sum:
                best_sum = s
                best_c = res.x[:2 * N].reshape(N, 2).copy()
                best_r = res.x[2 * N:].copy()
    except Exception:
        pass
        
    # Final LP verification to ensure radii match centers exactly
    lp_r, lp_s, _ = solve_lp_and_grad(best_c)
    if lp_s > best_sum:
        best_r = lp_r
        best_sum = lp_s
        
    final_radii = repair(best_c.copy(), best_r.copy())
    return best_c, final_radii, float(np.sum(final_radii))
