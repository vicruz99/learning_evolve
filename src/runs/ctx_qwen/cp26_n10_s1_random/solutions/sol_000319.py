# sol_000319 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000276 (state cc798eee) state=43d43ebb sum of radii=2.399999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute pair indices for efficiency
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
M_PAIRS = len(PAIR_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    A_ub = np.zeros((M_PAIRS, n))
    A_ub[np.arange(M_PAIRS), PAIR_I] = 1.0
    A_ub[np.arange(M_PAIRS), PAIR_J] = 1.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub = dists[PAIR_I, PAIR_J]
    
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, lim) for lim in lims]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def get_lp_grad(centers, res):
    """Computes gradient of LP sum of radii w.r.t centers using dual variables."""
    n = centers.shape[0]
    grad = np.zeros((n, 2))
    
    marg = None
    try:
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = np.asarray(res.ineqlin.marginals)
    except Exception:
        pass
        
    if marg is not None and len(marg) > 0:
        lams = marg[:M_PAIRS]
        valid = lams > 1e-8
        if np.any(valid):
            k = np.where(valid)[0]
            i_idx = PAIR_I[k]
            j_idx = PAIR_J[k]
            lam_vals = lams[k]
            
            diff = centers[i_idx] - centers[j_idx]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            dist_safe = np.maximum(dist, 1e-12)
            
            factors = (lam_vals / dist_safe)[:, np.newaxis]
            forces = diff * factors
            
            np.add.at(grad, i_idx, forces)
            np.add.at(grad, j_idx, -forces)
            
        # Boundary contributions
        for i in range(n):
            grad[i, 0] += marg[M_PAIRS + i] - marg[M_PAIRS + N + i]
            grad[i, 1] += marg[M_PAIRS + 2*N + i] - marg[M_PAIRS + 3*N + i]
            
    return grad

def obj_lp(x_flat):
    """Objective and gradient for L-BFGS-B."""
    c = x_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(x_flat)
    g = get_lp_grad(c, res)
    return -s, -g.flatten()

def obj_joint(v):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Inequality constraints >= 0 for SLSQP."""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    
    bc = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[PAIR_I] - cx[PAIR_J]
    dy = cy[PAIR_I] - cy[PAIR_J]
    d2 = dx**2 + dy**2
    rs = r[PAIR_I] + r[PAIR_J]
    pc = d2 - rs**2
    
    return np.concatenate([bc, pc])

def generate_configs(rng):
    """Generates diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal patterns
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [5,7,5,5,4], [4,5,6,5,6], [6,4,5,6,5]
    ]
    for pat in patterns:
        if sum(pat) < N: continue
        pts = []
        r0 = 0.101
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        base = np.array(pts[:N])
        configs.append(base)
        
        # Perturbations
        for _ in range(3):
            p = base + rng.uniform(-0.02, 0.02, base.shape)
            configs.append(np.clip(p, 0.03, 0.97))
            
    # 2. Grid patterns
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    grid = np.array([(x, y) for y in gy for x in gx])
    grid = np.vstack([grid, [0.5, 0.5]])[:N]
    configs.append(grid)
    
    # 3. Random dense
    for _ in range(10):
        configs.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    configs = generate_configs(rng)
    
    best_s = -1.0
    best_c = None
    best_r = None
    
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    # Phase 1: L-BFGS-B ascent on centers
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_lp, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Local coordinate descent / jiggle search
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.012
        
        for it in range(3000):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            
            # Random perturbation
            move = rng.uniform(-step, step, 2)
            new_c = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_c
            
            r_try, s_try, _ = solve_lp(curr_c)
            if r_try is not None and s_try > curr_s + 1e-9:
                curr_s = s_try
                curr_r = r_try.copy()
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step *= 0.999
            else:
                curr_c[idx] = old
                if rng.random() < 0.03:
                    step *= 0.94
                    
        # Phase 3: Joint SLSQP polish from refined centers
        for _ in range(5):
            c_pert = np.clip(best_c + rng.uniform(-0.003, 0.003, best_c.shape), 1e-4, 1.0-1e-4)
            r_pert = best_r * 0.995
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
            
            try:
                res_j = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints={'type': 'ineq', 'fun': cons_joint},
                                 options={'maxiter': 5000, 'ftol': 1e-14})
                if np.isfinite(res_j.fun):
                    c_j = np.column_stack((res_j.x[:N], res_j.x[N:2*N]))
                    r_j, s_j, _ = solve_lp(c_j)
                    if r_j is not None and s_j > best_s:
                        best_s = s_j
                        best_c = c_j.copy()
                        best_r = r_j.copy()
            except Exception:
                continue
                
    # Fallback safety net
    if best_c is None:
        best_c = np.tile(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)
        best_c = np.hstack([best_c, np.repeat(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)])
        best_c = np.vstack([best_c, [[0.5, 0.5]]])
        best_r, best_s, _ = solve_lp(best_c)
        
    # Final strict safety scaling to guarantee numerical validity
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
                
    best_r *= scale * 0.9999995
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
