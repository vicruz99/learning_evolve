# sol_000328 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000308 (state b85a4809) state=123f1dcf sum of radii=2.621753 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
M_PAIRS = len(PAIR_I)
M_CON = M_PAIRS + 4 * N

# Precompute LP constraint matrix structure
A_LP = np.zeros((M_CON, N))
for k, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
    A_LP[k, i] = 1.0
    A_LP[k, j] = 1.0
for i in range(N):
    A_LP[M_PAIRS + i, i] = 1.0
    A_LP[M_PAIRS + N + i, i] = 1.0
    A_LP[M_PAIRS + 2*N + i, i] = 1.0
    A_LP[M_PAIRS + 3*N + i, i] = 1.0

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii, sum, and solver result."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    b_ub = np.zeros(M_CON)
    diffs = centers[PAIR_I] - centers[PAIR_J]
    b_ub[:M_PAIRS] = np.hypot(diffs[:, 0], diffs[:, 1])
    
    for i in range(N):
        x, y = centers[i]
        b_ub[M_PAIRS + i] = x
        b_ub[M_PAIRS + N + i] = 1.0 - x
        b_ub[M_PAIRS + 2*N + i] = y
        b_ub[M_PAIRS + 3*N + i] = 1.0 - y
        
    bounds = [(0.0, lim) for lim in lims]
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def get_lp_grad(c_flat):
    """Objective and gradient for center optimization via LP duals."""
    c = c_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
        
    grad = np.zeros((N, 2))
    marg = None
    try:
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = np.asarray(res.ineqlin.marginals)
    except Exception:
        pass
        
    if marg is not None:
        # Pairwise constraints contribution
        lams = marg[:M_PAIRS]
        mask = lams > 1e-9
        if np.any(mask):
            idx = np.where(mask)[0]
            lam_vals = lams[idx]
            ii = PAIR_I[idx]
            jj = PAIR_J[idx]
            dx = c[ii, 0] - c[jj, 0]
            dy = c[ii, 1] - c[jj, 1]
            d = np.sqrt(dx**2 + dy**2)
            d = np.where(d < 1e-12, 1e-12, d)
            fx = lam_vals * dx / d
            fy = lam_vals * dy / d
            
            np.add.at(grad[:, 0], ii, fx)
            np.add.at(grad[:, 1], ii, fy)
            np.add.at(grad[:, 0], jj, -fx)
            np.add.at(grad[:, 1], jj, -fy)
            
        # Boundary constraints contribution
        for i in range(N):
            grad[i, 0] += marg[M_PAIRS + i] - marg[M_PAIRS + N + i]
            grad[i, 1] += marg[M_PAIRS + 2*N + i] - marg[M_PAIRS + 3*N + i]
            
    return -s, -grad.flatten()

def joint_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def joint_cons(v):
    """Inequality constraints >= 0 for SLSQP."""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    bc = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise non-overlap: d^2 >= (r_i + r_j)^2
    dx = cx[PAIR_I] - cx[PAIR_J]
    dy = cy[PAIR_I] - cy[PAIR_J]
    d2 = dx**2 + dy**2
    rs = r[PAIR_I] + r[PAIR_J]
    pc = d2 - rs**2
    
    return np.concatenate([bc, pc])

def generate_hex_configs(n, patterns, rng):
    """Generates diverse initial configurations based on hexagonal lattices."""
    configs = []
    for pat in patterns:
        if sum(pat) < n: continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
        base = np.array(pts[:n])
        
        # Normalize to fit comfortably in [0.05, 0.95]
        mn = base.min(axis=0)
        mx = base.max(axis=0)
        span = mx - mn + 1e-9
        norm_base = (base - mn) / span * 0.85 + 0.075
        configs.append(norm_base)
        
        # Add perturbations
        for _ in range(4):
            p = norm_base + rng.uniform(-0.02, 0.02, norm_base.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    # Add rotated versions to explore different alignments with square boundaries
    for angle in [0.15, 0.25, 0.35]:
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for cfg in configs[::6]:
            centered = cfg - 0.5
            rotated = (rot @ centered.T).T + 0.5
            configs.append(np.clip(rotated, 0.05, 0.95))
            
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    # 1. Generate diverse initial configurations
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], [4,6,6,6,4],
        [5,7,5,5,4], [6,5,5,6,4], [5,6,6,4,5], [7,6,6,7], [8,6,6,6],
        [5,5,5,5,6], [6,5,5,5,5], [5,5,5,6,5], [5,5,6,6,4], [5,6,5,5,5],
        [6,4,6,6,4], [4,6,5,6,5], [5,6,4,6,5], [6,5,6,4,5], [7,5,6,5,3]
    ]
    
    configs = generate_hex_configs(N, patterns, rng)
    
    # Add random dense starts
    for _ in range(10):
        configs.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(get_lp_grad, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Joint SLSQP polish for precise constraint handling
    if best_c is not None:
        for _ in range(8):
            c_pert = np.clip(best_c + rng.uniform(-0.005, 0.005, best_c.shape), 1e-4, 1.0-1e-4)
            r_pert = best_r * 0.99
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
            
            try:
                res_j = minimize(joint_obj, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints={'type': 'ineq', 'fun': joint_cons},
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
                
    # Phase 3: Simulated Annealing / Iterative Perturbation to escape local minima
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.012
        
        for it in range(1500):
            # Adaptive step size decay
            step = max(0.001, step * 0.998)
            
            # Perturb a random subset of circles
            n_perturb = rng.integers(3, 8)
            idxs = rng.choice(N, n_perturb, replace=False)
            old = curr_c[idxs].copy()
            
            move = rng.uniform(-step, step, (n_perturb, 2))
            curr_c[idxs] = np.clip(curr_c[idxs] + move, 1e-4, 1.0 - 1e-4)
            
            r_try, s_try, _ = solve_lp(curr_c)
            if r_try is not None and s_try > curr_s + 1e-8:
                curr_s = s_try
                curr_r = r_try.copy()
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step *= 1.02 # Increase step on success
            else:
                curr_c[idxs] = old # Revert on failure
                
    # Fallback safety net
    if best_c is None:
        best_c = np.clip(configs[0], 1e-4, 1.0-1e-4)
        best_r, best_s, _ = solve_lp(best_c)
        
    # Final strict safety scaling to guarantee numerical validity against 1e-12 tolerance
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
