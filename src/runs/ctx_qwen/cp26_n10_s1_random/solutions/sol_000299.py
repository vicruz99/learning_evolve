# sol_000299 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000293 (state 498c5d88) state=3e7613e7 sum of radii=2.630699 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)
M_PAIRS = len(TRIU_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    c_obj = -np.ones(n)
    A_ub = np.zeros((M_PAIRS, n))
    A_ub[np.arange(M_PAIRS), TRIU_I] = 1.0
    A_ub[np.arange(M_PAIRS), TRIU_J] = 1.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub = dists[TRIU_I, TRIU_J]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0, None

def get_lp_gradient(centers, lp_res):
    """Computes gradient of LP objective w.r.t centers using dual variables."""
    n = centers.shape[0]
    grad = np.zeros((n, 2))
    if lp_res is None:
        return grad
        
    try:
        # Handle different scipy versions for marginal attributes
        marginals = getattr(getattr(lp_res, 'ineqlin', None), 'marginals', None)
        if marginals is None:
            marginals = getattr(getattr(lp_res, 'marginals', None), 'ineqlin', None)
        if marginals is None:
            return grad
            
        gains = np.maximum(marginals[:M_PAIRS], 0.0)
        
        if np.any(gains > 1e-8):
            valid = gains > 1e-8
            ii = TRIU_I[valid]
            jj = TRIU_J[valid]
            g = gains[valid]
            d_ij = np.sqrt(np.sum((centers[ii] - centers[jj])**2, axis=1))
            d_safe = np.where(d_ij > 1e-8, d_ij, 1.0)
            vecs = (g / d_safe)[:, np.newaxis] * (centers[ii] - centers[jj])
            np.add.at(grad, ii, vecs)
            np.add.at(grad, jj, -vecs)
    except Exception:
        pass
    return grad

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
    rng = np.random.default_rng(2025)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,5,6,2], [8,6,5,5,2], [9,5,5,5,2]
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
        # Normalize to fit comfortably inside
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn + 1e-9
        pts = (pts - mn) / span * 0.72 + 0.14
        inits.append(pts)
        
        for _ in range(3):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            inits.append(np.clip(p, 0.05, 0.95))
            
    # Corner-biased and random inits
    for _ in range(6):
        pts = rng.uniform(0.05, 0.95, (N, 2))
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        for i, c in enumerate(corners):
            pts[i] = c + rng.uniform(-0.04, 0.04, 2)
        inits.append(np.clip(pts, 0.02, 0.98))
        
    for _ in range(8):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_vars = [(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N

    # Phase 1: Initial LP refinement + SLSQP
    for cfg in inits:
        r0, s0, _ = solve_lp(cfg)
        x0 = np.concatenate([cfg[:,0], cfg[:,1], r0 * 0.99])
        try:
            res = minimize(joint_obj, x0, method='SLSQP', bounds=bounds_vars,
                           constraints={'type': 'ineq', 'fun': joint_cons},
                           options={'maxiter': 4000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_mat = np.column_stack((res.x[:N], res.x[N:2*N]))
                r_lp, s_lp, _ = solve_lp(c_mat)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_mat.copy()
                    best_radii = r_lp.copy()
        except: pass

    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum, _ = solve_lp(best_centers)

    # Phase 2: LP Gradient Ascent with Momentum
    c = best_centers.copy()
    r = best_radii.copy()
    lr = 0.006
    vel = np.zeros_like(c)
    
    for step in range(600):
        r, s, lp_res = solve_lp(c)
        grad = get_lp_gradient(c, lp_res)
        
        vel = 0.5 * vel + lr * grad
        c_new = np.clip(c + vel, 1e-4, 1.0 - 1e-4)
        
        r_new, s_new, _ = solve_lp(c_new)
        if s_new > s + 1e-7:
            c = c_new
            r = r_new
            lr = min(lr * 1.015, 0.025)
        else:
            lr *= 0.92
            vel *= 0.4
            
        if s_new > best_sum:
            best_sum = s_new
            best_centers = c.copy()
            best_radii = r.copy()

    # Phase 3: Coordinate Descent with Simulated Annealing
    curr_c = best_centers.copy()
    curr_r, curr_s, _ = solve_lp(curr_c)
    
    temperature = 0.04
    for epoch in range(15):
        improved = False
        for i in range(N):
            old = curr_c[i].copy()
            best_move = old
            best_move_s = curr_s
            
            step_sizes = np.logspace(-3, -1.5, 4) * (0.85 ** epoch)
            for ss in step_sizes:
                for _ in range(8):
                    d = rng.uniform(-ss, ss, 2)
                    new_pos = np.clip(curr_c[i] + d, 1e-4, 0.999)
                    curr_c[i] = new_pos
                    r_try, s_try, _ = solve_lp(curr_c)
                    
                    delta = s_try - curr_s
                    if delta > 0 or rng.random() < np.exp(delta / temperature):
                        if s_try > best_move_s:
                            best_move_s = s_try
                            best_move = new_pos.copy()
                    curr_c[i] = old
                    
            curr_c[i] = best_move
            curr_r, curr_s, _ = solve_lp(curr_c)
            if curr_s > best_sum + 1e-8:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                improved = True
                
        temperature *= 0.82
        if not improved:
            break
            
    # Phase 4: Final SLSQP Polish
    x0 = np.concatenate([best_centers[:,0], best_centers[:,1], best_radii * 0.995])
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
    except: pass

    # Phase 5: Strict Safety Scaling
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
