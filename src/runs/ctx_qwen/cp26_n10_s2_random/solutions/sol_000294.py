# sol_000294 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000243 (state e183a9b7) state=d3e3b453 sum of radii=2.598971 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure globally
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

def l_bfgs_b_obj_grad(v):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
    _, s, g = solve_lp_and_grad(centers)
    return -s, -g.flatten()

def slsqp_obj(v):
    """Objective for joint SLSQP optimization."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Constraints for joint SLSQP optimization."""
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

def slsqp_optimize(c0, r0, maxiter=5000):
    """Runs joint SLSQP optimization of centers and radii."""
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    try:
        res = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': slsqp_cons},
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'disp': False})
        c_vals = slsqp_cons(res.x)
        if np.min(c_vals) >= -1e-7:
            return res.x[:2*N].reshape(N, 2), res.x[2*N:], -res.fun
    except Exception:
        pass
    return c0, r0, np.sum(r0)

def coord_obj(p, centers, i):
    """Objective for coordinate-wise optimization of a single circle's position."""
    cx, cy = p
    c_tmp = centers.copy()
    c_tmp[i] = np.clip([cx, cy], 1e-5, 1.0 - 1e-5)
    _, s, _ = solve_lp_and_grad(c_tmp)
    return -s

def generate_starts(rng):
    """Generates a diverse set of initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]]
    for pat in pats:
        for r0 in [0.092, 0.098, 0.105, 0.110]:
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
            
    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.2 and d > 1e-4:
                        push = (0.2 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f
            c = np.clip(c, 0.05, 0.95)
        starts.append(c)
        
    for _ in range(10):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
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
    bounds_xy = [(1e-5, 0.995)] * (2 * N)
    
    # Phase 1: L-BFGS-B on centers from multiple diverse starts
    for c_init in starts:
        try:
            res = minimize(l_bfgs_b_obj_grad, c_init.flatten(), jac=True, method='L-BFGS-B', 
                           bounds=bounds_xy, options={'maxiter': 2000, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Systematic perturbations & L-BFGS-B restarts to escape local minima
    for _ in range(25):
        c_pert = best_c.copy()
        c_pert += rng.normal(0, 0.008, c_pert.shape)
        c_pert = np.clip(c_pert, 0.05, 0.95)
        try:
            res = minimize(l_bfgs_b_obj_grad, c_pert.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_xy, options={'maxiter': 1500, 'ftol': 1e-13})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP Joint Polish from perturbed neighborhoods
    for _ in range(5):
        v_pert = best_c.flatten().copy()
        v_pert += rng.normal(0, 0.003, v_pert.shape)
        v_pert = np.clip(v_pert, 0.01, 0.99)
        c_pert = v_pert.reshape(N, 2)
        r_pert, _, _ = solve_lp_and_grad(c_pert)
        
        c_sl, r_sl, s_sl = slsqp_optimize(c_pert, r_pert, maxiter=6000)
        if s_sl > best_s:
            best_s = s_sl
            best_c = c_sl.copy()
            best_r = r_sl.copy()
            
    # Phase 4: Coordinate-wise local optimization to squeeze marginal gains
    for _ in range(3):
        for i in range(N):
            try:
                res = minimize(coord_obj, best_c[i], args=(best_c, i), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-10})
                best_c[i] = np.clip(res.x, 1e-5, 1.0 - 1e-5)
            except Exception:
                pass
        _, s_new, _ = solve_lp_and_grad(best_c)
        if s_new > best_s:
            best_s = s_new
            best_r = solve_lp_and_grad(best_c)[0]
            
    # Final rigorous SLSQP polish
    c_final, r_final, s_final = slsqp_optimize(best_c, best_r, maxiter=8000)
    if s_final > best_s:
        best_c = c_final
        best_r = r_final
        best_s = s_final
        
    # Strict numerical repair to guarantee validation passes
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
