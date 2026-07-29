# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000143 (state b4622c9f) state=b054ac58 sum of radii=2.174654 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def get_lp_and_duals(centers):
    """Solves LP for radii and returns radii, pairwise dual variables, and distances."""
    n = centers.shape[0]
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((4 * n + num_pairs, n))
    b_ub = np.zeros(4 * n + num_pairs)
    k = 0
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        bounds_val = [x, 1.0 - x, y, 1.0 - y]
        for b in bounds_val:
            A_ub[k, i] = 1.0
            b_ub[k] = b
            k += 1
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    c_obj = -np.ones(n)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Extract pairwise marginals (dual variables)
            # They indicate how much objective improves per unit increase in distance
            marginals = np.asarray(res.ineqlin.marginals).ravel()
            pair_marginals = marginals[4 * n:]
            return res.x, pair_marginals, dists
    except Exception:
        pass
        
    # Fallback if LP fails
    return np.full(n, 1e-6), np.zeros(num_pairs), dists

def compute_center_gradient(centers, pair_marginals, dists):
    """Computes gradient of sum(radii) w.r.t centers using LP dual variables."""
    n = centers.shape[0]
    grad = np.zeros((n, 2))
    pair_idx = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 1e-12:
                continue
            marginal = pair_marginals[pair_idx]
            if marginal > 1e-10:
                diff = centers[i] - centers[j]
                force = marginal * diff / d
                grad[i] += force
                grad[j] -= force
            pair_idx += 1
    return grad

def gradient_ascent_search(centers, max_iter=600, init_step=0.04):
    """Performs gradient ascent on centers to maximize sum of radii."""
    best_centers = centers.copy()
    r, marginals, dists = get_lp_and_duals(centers)
    best_sum = np.sum(r)
    step = init_step
    
    for _ in range(max_iter):
        grad = compute_center_gradient(centers, marginals, dists)
        norm_g = np.linalg.norm(grad)
        if norm_g < 1e-14:
            break
            
        new_centers = centers + step * grad
        # Strictly keep inside square with small margin
        new_centers = np.clip(new_centers, 0.008, 0.992)
        
        r_new, marginals_new, dists_new = get_lp_and_duals(new_centers)
        new_sum = np.sum(r_new)
        
        if new_sum > best_sum + 1e-12:
            centers = new_centers
            best_sum = new_sum
            best_centers = centers.copy()
            marginals = marginals_new
            dists = dists_new
            step = min(step * 1.08, 0.25)
        else:
            step *= 0.85
            
    return best_centers, best_sum

def hex_init(rng, pattern, rot, scale):
    pts = []
    r_est = 0.098
    y = r_est
    for r_idx, cnt in enumerate(pattern):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        M = np.array([[c, -s], [s, c]])
        pts = (pts - 0.5) @ M.T + 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(rng):
    pts = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(400):
        f = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist = np.maximum(dist, 1e-4)
        rep = 1.0 / (dist**2)
        np.fill_diagonal(rep, 0.0)
        f += np.sum(rep[:, :, None] * diff / dist[:, :, None], axis=1)
        for d in range(2):
            f[:, d] += 25.0 * np.maximum(0, 0.08 - pts[:, d])
            f[:, d] -= 25.0 * np.maximum(0, pts[:, d] - 0.92)
        pts += 0.005 * f
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def to_params(centers, radii=None):
    if radii is None:
        radii, _, _ = get_lp_and_duals(centers)
    r = np.maximum(radii * 0.992, 1e-6)
    x, y = centers[:, 0], centers[:, 1]
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((x - r) / denom, 0.0, 1.0)
    v = np.clip((y - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def from_params(params):
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack([x, y]), r

def constraints_slqp(params):
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def objective_slqp(params):
    return -np.sum(params[:N])

def run_packing():
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons_slqp = {'type': 'ineq', 'fun': constraints_slqp}
    
    best_c = None
    best_sum = -np.inf
    
    # Generate diverse initial configurations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6], [6,4,6,5,5],
        [7,5,5,5,4], [5,7,5,5,4], [4,5,5,5,7], [6,6,5,5,4], [5,5,6,5,5],
        [6,5,5,5,5], [5,5,5,6,5], [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3]
    ]
    idx = 0
    for p in patterns:
        for _ in range(3):
            rot = rng.uniform(-0.12, 0.12)
            sc = rng.uniform(0.90, 1.10)
            inits.append(hex_init(rng, p, rot, sc))
            idx += 1
    for _ in range(12):
        inits.append(force_init(rng))
        
    # Phase 1: Gradient Ascent from diverse starts
    for pts in inits:
        c_opt, s_opt = gradient_ascent_search(pts)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            
    # Phase 2: Topological escape via jittered restarts
    if best_c is not None:
        curr_c = best_c.copy()
        curr_sum = best_sum
        for k in range(40):
            # Perturb best solution to escape local contact graphs
            noise = rng.normal(0, 0.008 * (1.0 + 0.5*np.exp(-k/10.0)), (N, 2))
            perturbed = np.clip(curr_c + noise, 0.02, 0.98)
            c_opt, s_opt = gradient_ascent_search(perturbed, max_iter=300, init_step=0.02)
            if s_opt > curr_sum:
                curr_c = c_opt.copy()
                curr_sum = s_opt
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_c = curr_c.copy()
                    
    # Phase 3: High-precision SLSQP joint optimization
    if best_c is not None:
        x0 = to_params(best_c)
        try:
            res = minimize(objective_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if res.success and np.min(constraints_slqp(res.x)) >= -1e-9:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_c, _ = from_params(res.x)
        except Exception:
            pass
            
        # Final ultra-fine polish
        x0 = to_params(best_c)
        try:
            res = minimize(objective_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 8000, 'ftol': 1e-15, 'disp': False})
            if res.success and np.min(constraints_slqp(res.x)) >= -1e-10:
                s_val = -res.fun
                if s_val > best_sum:
                    best_c, _ = from_params(res.x)
        except Exception:
            pass

    # Fallback (should not be reached)
    if best_c is None:
        best_c = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                  np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        r_f, _, _ = get_lp_and_duals(best_c)
        best_c, best_sum = best_c, np.sum(r_f)
        
    # Recompute exact optimal radii for returned centers
    final_r, _, _ = get_lp_and_duals(best_c)
    final_r = np.maximum(final_r, 0.0)
    
    return best_c, final_r, float(np.sum(final_r))
