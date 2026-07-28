# sol_000296 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000293 (state 498c5d88) state=dc3a5358 sum of radii=2.614089 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)
M = len(TRIU_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub = dists[TRIU_I, TRIU_J]
    
    A = np.zeros((M, N))
    A[np.arange(M), TRIU_I] = 1.0
    A[np.arange(M), TRIU_J] = 1.0
    
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return np.full(N, 1e-9), 0.0, None

def gradient_ascent(c0, max_iter=150):
    """LP-guided gradient ascent on circle centers."""
    c = c0.copy()
    best_c, best_r, best_s = c0.copy(), np.full(N, 1e-9), -1.0
    lr = 0.006
    vel = np.zeros_like(c)
    
    for _ in range(max_iter):
        r, s, res = solve_lp(c)
        if r is None: break
        if s > best_s:
            best_s = s
            best_c = c.copy()
            best_r = r.copy()
            
        grad = np.zeros_like(c)
        try:
            marg = res.marginals.ineqlin
            if hasattr(marg, 'toarray'): marg = marg.toarray().flatten()
            # Marginals for min -sum(r) are <= 0. Gain per distance increase is -marg.
            w = np.maximum(-marg, 0.0)
            if np.any(w > 1e-8):
                d = np.linalg.norm(c[TRIU_I] - c[TRIU_J], axis=1)
                d_safe = np.where(d > 1e-9, d, 1.0)
                vec = (w / d_safe)[:, np.newaxis] * (c[TRIU_I] - c[TRIU_J])
                np.add.at(grad, TRIU_I, vec)
                np.add.at(grad, TRIU_J, -vec)
        except Exception:
            pass
            
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-7:
            vel = 0.7 * vel + lr * (grad / g_norm)
            c = c + vel
            c = np.clip(c, 1e-5, 1.0 - 1e-5)
            lr *= 0.985
        else:
            break
    return best_c, best_r, best_s

def joint_obj(v):
    """Objective for joint SLSQP optimization."""
    return -np.sum(v[2*N:])

def joint_cons(v):
    """Inequality constraints >= 0 for joint optimization."""
    cx, cy, r = v[:N], v[N:2*N], v[2*N:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[TRIU_I] - cx[TRIU_J]
    dy = cy[TRIU_I] - cy[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,5,5,5,4], [8,6,6,6], [9,5,6,6], [10,8,8]
    ]
    
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.095
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        base = np.array(pts[:N])
        inits.append(base)
        
        for _ in range(2):
            p = base + rng.uniform(-0.025, 0.025, base.shape)
            inits.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(8):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    return inits

def run_packing() -> tuple:
    rng = np.random.default_rng(42)
    best_c, best_r, best_s = None, None, -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: LP-gradient ascent from diverse starts
    for cfg in inits:
        c_opt, r_opt, s_opt = gradient_ascent(cfg, max_iter=120)
        if s_opt > best_s:
            best_s = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    # Phase 2: Basin-hopping perturbations to escape local minima
    if best_c is not None:
        for _ in range(25):
            pert = best_c + rng.uniform(-0.005, 0.005, best_c.shape)
            pert = np.clip(pert, 0.02, 0.98)
            c_p, r_p, s_p = gradient_ascent(pert, max_iter=100)
            if s_p > best_s:
                best_s = s_p
                best_c = c_p.copy()
                best_r = r_p.copy()
                
    # Phase 3: Joint SLSQP polish for precise constraint handling
    if best_c is not None:
        x0 = np.zeros(3*N)
        x0[:N] = best_c[:, 0]
        x0[N:2*N] = best_c[:, 1]
        x0[2*N:] = best_r * 0.985
        
        bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
        try:
            res = minimize(joint_obj, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints={'type': 'ineq', 'fun': joint_cons},
                           options={'maxiter': 5000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_slq = np.column_stack((res.x[:N], res.x[N:2*N]))
                r_slq, s_slq, _ = solve_lp(c_slq)
                if s_slq > best_s:
                    best_s = s_slq
                    best_c = c_slq.copy()
                    best_r = r_slq.copy()
        except Exception:
            pass
            
    # Fallback
    if best_c is None:
        best_c = inits[0]
        best_r, best_s, _ = solve_lp(best_c)
        
    # Final safety scaling to guarantee strict numerical validity
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_r *= scale * 0.9999999
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
