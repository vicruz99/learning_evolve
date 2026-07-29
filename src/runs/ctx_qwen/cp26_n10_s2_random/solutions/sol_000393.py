# sol_000393 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000360 (state f30435ab) state=11a4b10a sum of radii=2.616034 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, 1)
NUM_PAIRS = N * (N - 1) // 2

# Precompute LP constraint matrix structure (constant across runs)
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
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-16)
    
    b_ub = np.zeros(NUM_PAIRS + 4 * N)
    k = 0
    for i, j in PAIR_IDX:
        b_ub[k] = dists[i, j]
        k += 1
    for i in range(N):
        b_ub[k] = centers[i, 0]; k += 1
        b_ub[k] = 1.0 - centers[i, 0]; k += 1
        b_ub[k] = centers[i, 1]; k += 1
        b_ub[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0.0, u) for u in ub], method='highs')
        if not res.success:
            return np.full(N, 0.05), np.sum(np.full(N, 0.05)), np.zeros_like(centers)
    except Exception:
        return np.full(N, 0.05), np.sum(np.full(N, 0.05)), np.zeros_like(centers)
        
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
        if mu > 1e-9:
            d = max(dists[i, j], 1e-9)
            vec = (centers[i] - centers[j]) / d
            grad[i] += mu * vec
            grad[j] -= mu * vec
        k += 1
        
    b_start = NUM_PAIRS
    for i in range(N):
        grad[i, 0] += duals[b_start + 4*i] - duals[b_start + 4*i + 1]
        grad[i, 1] += duals[b_start + 4*i + 2] - duals[b_start + 4*i + 3]
        
    return radii, s_sum, grad

def obj_grad_lp(x):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    c = x.reshape(N, 2)
    _, s, g = solve_lp_and_grad(c)
    return -s, -g.flatten()

def constraints_joint(v):
    """Computes boundary and non-overlap constraints for joint SLSQP optimization."""
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

def objective_joint(v):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def generate_hex_start(rng, pattern, r0, noise=0.005):
    """Generates a hexagonal lattice initialization."""
    c = []
    y = r0
    for ri, cnt in enumerate(pattern):
        sh = r0 if ri % 2 == 1 else 0.0
        x = r0 + sh
        for _ in range(cnt):
            if len(c) < N:
                c.append([x + rng.normal(0, noise), y + rng.normal(0, noise)])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
    return np.clip(np.array(c[:N]), 0.02, 0.98)

def generate_starts(rng):
    """Generates diverse initial configurations."""
    starts = []
    pats = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
            [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
            [6, 5, 5, 6, 4], [5, 6, 4, 5, 6], [6, 4, 5, 6, 5],
            [5, 5, 4, 6, 6], [4, 5, 6, 6, 5], [6, 5, 6, 4, 5]]
    for pat in pats:
        for r0 in [0.090, 0.095, 0.100, 0.105, 0.110]:
            starts.append(generate_hex_start(rng, pat, r0))
            
    for _ in range(10):
        starts.append(generate_hex_start(rng, pats[rng.integers(len(pats))], 0.10, noise=0.02))
        
    # Random
    for _ in range(15):
        starts.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Force directed
    for _ in range(10):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i] - c[j]
                    d = np.linalg.norm(dv)
                    if d < 0.25 and d > 1e-4:
                        push = (0.25 - d) * 0.05 / (d + 1e-4)
                        f[i] += dv / d * push
                        f[j] -= dv / d * push
            c += f * 0.02
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
    rng = np.random.default_rng(42)
    best_c = None
    best_r = None
    best_s = -1.0
    
    starts = generate_starts(rng)
    bounds_c = [(0.0001, 0.9999)] * (2 * N)
    bounds_joint = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Phase 1: L-BFGS-B on all starts
    for c_init in starts:
        try:
            res = minimize(obj_grad_lp, c_init.flatten(), jac=True, method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 5000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Momentum-based Gradient Ascent to escape flat regions
    c_mom = best_c.copy()
    v_mom = np.zeros_like(c_mom)
    step = 0.002
    for _ in range(800):
        _, _, grad = solve_lp_and_grad(c_mom)
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-8:
            direction = grad / g_norm
            v_mom = 0.75 * v_mom + 0.25 * step * direction
            c_new = np.clip(c_mom + v_mom, 0.001, 0.999)
            _, s_new, _ = solve_lp_and_grad(c_new)
            if s_new > best_s:
                c_mom = c_new
                best_s = s_new
                best_c = c_mom.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
                step = min(step * 1.05, 0.008)
            else:
                step *= 0.95
        else:
            step *= 0.95

    # Phase 3: Simulated Annealing with periodic local refinement
    c_curr = best_c.copy()
    s_curr = best_s
    T = 0.012
    for step_idx in range(2000):
        noise = T * 0.35
        c_try = c_curr + rng.normal(0, noise, c_curr.shape)
        c_try = np.clip(c_try, 0.02, 0.98)
        _, s_try, _ = solve_lp_and_grad(c_try)
        
        delta = s_try - s_curr
        if delta > 0 or rng.random() < np.exp(np.clip(delta / max(T, 1e-9), -20, 20)):
            c_curr, s_curr = c_try, s_try
            if s_curr > best_s:
                best_s = s_curr
                best_c = c_curr.copy()
                best_r, _, _ = solve_lp_and_grad(best_c)
        T *= 0.997
        
        # Occasional local gradient step to settle
        if step_idx % 60 == 0:
            try:
                res = minimize(obj_grad_lp, c_curr.flatten(), jac=True, method='L-BFGS-B',
                               bounds=bounds_c, options={'maxiter': 800, 'ftol': 1e-13})
                c_loc = res.x.reshape(N, 2)
                r_loc, s_loc, _ = solve_lp_and_grad(c_loc)
                if s_loc > s_curr:
                    c_curr, s_curr = c_loc, s_loc
                    if s_loc > best_s:
                        best_s = s_loc
                        best_c = c_curr.copy()
                        best_r = r_loc.copy()
            except Exception:
                pass
                
    # Phase 4: Coordinate-wise targeted perturbations
    for _ in range(30):
        c_k = best_c.copy()
        idx = rng.choice(N, size=rng.integers(3, 9), replace=False)
        c_k[idx] += rng.normal(0, 0.012, (len(idx), 2))
        c_k = np.clip(c_k, 0.02, 0.98)
        
        try:
            res = minimize(obj_grad_lp, c_k.flatten(), jac=True, method='L-BFGS-B',
                           bounds=bounds_c, options={'maxiter': 3000, 'ftol': 1e-14})
            c_opt = res.x.reshape(N, 2)
            r_opt, s_opt, _ = solve_lp_and_grad(c_opt)
            if s_opt > best_s:
                best_s = s_opt
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            pass

    # Phase 5: Joint SLSQP Polish for final precision
    try:
        v0 = np.concatenate([best_c.flatten(), best_r])
        res = minimize(objective_joint, v0, method='SLSQP', bounds=bounds_joint,
                       constraints={'type': 'ineq', 'fun': constraints_joint},
                       options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        if np.min(constraints_joint(res.x)) >= -1e-7:
            s_sl = np.sum(res.x[2*N:])
            if s_sl > best_s:
                best_s = s_sl
                best_c = res.x[:2*N].reshape(N, 2)
                best_r = res.x[2*N:]
    except Exception:
        pass
        
    # Final repair and validation compliance
    radii = repair(best_c, best_r)
    return best_c, radii, float(np.sum(radii))
