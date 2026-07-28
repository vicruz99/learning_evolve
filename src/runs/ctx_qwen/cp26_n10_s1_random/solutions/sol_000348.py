# sol_000348 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000299 (state 3e7613e7) state=3e8e1a07 sum of radii=0.144000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)
M_PAIRS = len(TRIU_I)
M_CON = M_PAIRS + 4 * N

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((M_CON, n))
    b_ub = np.zeros(M_CON)
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub[:M_PAIRS] = dists[TRIU_I, TRIU_J]
    A_ub[:M_PAIRS, TRIU_I] = 1.0
    A_ub[:M_PAIRS, TRIU_J] = 1.0
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        A_ub[M_PAIRS + i, i] = 1.0
        b_ub[M_PAIRS + i] = centers[i, 0]
        
        A_ub[M_PAIRS + N + i, i] = 1.0
        b_ub[M_PAIRS + N + i] = 1.0 - centers[i, 0]
        
        A_ub[M_PAIRS + 2*N + i, i] = 1.0
        b_ub[M_PAIRS + 2*N + i] = centers[i, 1]
        
        A_ub[M_PAIRS + 3*N + i, i] = 1.0
        b_ub[M_PAIRS + 3*N + i] = 1.0 - centers[i, 1]
        
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0, None

def lp_obj_grad(c_flat):
    """Objective and exact gradient for center optimization via LP duals."""
    c = c_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 0.0, np.zeros_like(c_flat)
        
    grad = np.zeros((N, 2))
    try:
        marg = None
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = np.asarray(res.ineqlin.marginals)
            
        if marg is not None and len(marg) == M_CON:
            # Pairwise constraints contribution
            lams = np.maximum(marg[:M_PAIRS], 0.0)
            d_ij = np.sqrt(np.sum((c[TRIU_I] - c[TRIU_J])**2, axis=1))
            d_safe = np.where(d_ij > 1e-9, d_ij, 1.0)
            vecs = (lams / d_safe)[:, np.newaxis] * (c[TRIU_I] - c[TRIU_J])
            np.add.at(grad[:, 0], TRIU_I, vecs[:, 0])
            np.add.at(grad[:, 1], TRIU_I, vecs[:, 1])
            np.add.at(grad[:, 0], TRIU_J, -vecs[:, 0])
            np.add.at(grad[:, 1], TRIU_J, -vecs[:, 1])
            
            # Boundary constraints contribution
            for i in range(N):
                grad[i, 0] += np.maximum(marg[M_PAIRS + i], 0.0) - np.maximum(marg[M_PAIRS + N + i], 0.0)
                grad[i, 1] += np.maximum(marg[M_PAIRS + 2*N + i], 0.0) - np.maximum(marg[M_PAIRS + 3*N + i], 0.0)
    except Exception:
        pass
        
    return -s, -grad.flatten()

def joint_obj(v):
    return -np.sum(v[2*N:])

def joint_cons(v):
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    dx = x[TRIU_I] - x[TRIU_J]
    dy = y[TRIU_I] - y[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_c = [(1e-4, 0.999)] * (2 * N)
    
    # Generate diverse initial configurations
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,5,6,2], [8,6,5,5,2], [9,5,5,5,2], [5,7,4,5,5]
    ]
    
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.095
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx%2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:N])
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn + 1e-9
        pts = (pts - mn) / span * 0.72 + 0.14
        configs.append(pts)
        
        for _ in range(2):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(5):
        pts = rng.uniform(0.1, 0.9, (N, 2))
        configs.append(pts)
        
    # Phase 1: L-BFGS-B with exact LP gradient
    for cfg in configs:
        x0 = np.clip(cfg.flatten(), 1e-4, 0.999)
        try:
            res = minimize(lp_obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Phase 2: Basin hopping / Perturbation around best
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for epoch in range(25):
            improved = False
            shift_mag = 0.025 * (0.96 ** epoch)
            pert = curr_c + rng.uniform(-shift_mag, shift_mag, (N, 2))
            pert = np.clip(pert, 1e-4, 0.999)
            
            x0 = pert.flatten()
            try:
                res = minimize(lp_obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds_c,
                               options={'maxiter': 2000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    c_opt = res.x.reshape(N, 2)
                    r_opt, s_opt, _ = solve_lp(c_opt)
                    if s_opt > curr_s + 1e-7:
                        curr_s = s_opt
                        curr_c = c_opt.copy()
                        curr_r = r_opt.copy()
                        improved = True
            except Exception:
                pass
                
            if not improved:
                for _ in range(12):
                    idx = rng.integers(N)
                    old = curr_c[idx].copy()
                    curr_c[idx] = np.clip(curr_c[idx] + rng.uniform(-0.03, 0.03, 2), 1e-4, 0.999)
                    r_try, s_try, _ = solve_lp(curr_c)
                    if s_try > curr_s + 1e-7:
                        curr_s = s_try
                        curr_r = r_try.copy()
                    else:
                        curr_c[idx] = old
                        
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()

    # Phase 3: SLSQP Joint Polish
    if best_centers is not None:
        x0 = np.concatenate([best_centers[:,0], best_centers[:,1], best_radii * 0.995])
        bounds_vars = [(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N
        try:
            res = minimize(joint_obj, x0, method='SLSQP', bounds=bounds_vars,
                           constraints={'type': 'ineq', 'fun': joint_cons},
                           options={'maxiter': 6000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_mat = np.column_stack((res.x[:N], res.x[N:2*N]))
                r_lp, s_lp, _ = solve_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Fallback
    if best_centers is None:
        best_centers = np.tile(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)
        best_centers = np.hstack([best_centers, np.repeat(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)])
        best_centers = np.vstack([best_centers, [[0.5, 0.5]]])
        best_radii, best_sum, _ = solve_lp(best_centers)

    # Safety Scaling
    scale = 1.0
    for i in range(N):
        x, y, r = best_centers[i,0], best_centers[i,1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
