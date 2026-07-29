# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000057 (state 347516f5) state=bc8e03df sum of radii=2.619018 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    try:
        n = centers.shape[0]
        c_obj = -np.ones(n)
        num_c = 4 * n + n * (n - 1) // 2
        A_ub = np.zeros((num_c, n))
        b_ub = np.zeros(num_c)
        k = 0
        
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(n):
            x, y = centers[i]
            for b in (x, 1.0 - x, y, 1.0 - y):
                A_ub[k, i] = 1.0
                b_ub[k] = b
                k += 1
                
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
        dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        for i in range(n):
            for j in range(i + 1, n):
                A_ub[k, i] = 1.0
                A_ub[k, j] = 1.0
                b_ub[k] = dists[i, j]
                k += 1
                
        bounds = [(0.0, None)] * n
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def to_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def from_params(params):
    """Reconstruct physical centers and radii from (r, u, v) parameters."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack([x, y]), r

def cons_slqp(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def obj_slqp(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def hex_init(rng, row_counts, rot, scale):
    """Generates a hexagonal lattice initialization."""
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    for cnt in row_counts:
        shift = (row % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if abs(rot) > 1e-6:
        c, s = np.cos(rot), np.sin(rot)
        M = np.array([[c, -s], [s, c]])
        pts = (pts - 0.5) @ M.T + 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(rng):
    """Generates a force-directed layout initialization."""
    pts = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(400):
        f = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        d = np.maximum(d, 1e-4)
        f += np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
        for dim in range(2):
            f[:, dim] += 15.0 * np.maximum(0, 0.1 - pts[:, dim])
            f[:, dim] -= 15.0 * np.maximum(0, pts[:, dim] - 0.9)
        pts += 0.002 * f
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def grid_init(rng):
    """Generates a perturbed grid initialization."""
    pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    pts = np.vstack([pts, [0.5, 0.5]])
    pts += rng.uniform(-0.02, 0.02, (N, 2))
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_c = None
    best_r = None
    best_sum = -np.inf
    
    inits = []
    pats = [[6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [7,5,5,5,4], [5,5,5,5,6], 
            [6,6,5,5,4], [5,5,6,5,5], [6,5,5,5,5], [5,5,5,6,5], [6,4,6,5,5],
            [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3]]
    for pat in pats:
        for _ in range(3):
            inits.append(hex_init(rng, pat, rot=rng.uniform(-0.3, 0.3), scale=rng.uniform(0.85, 1.15)))
    for _ in range(10):
        inits.append(force_init(rng))
    for _ in range(8):
        inits.append(grid_init(rng))
        
    # Evaluate initial configurations with LP
    for c in inits:
        r, s_val = solve_lp_radii(c)
        if s_val > best_sum:
            best_sum = s_val
            best_c = c.copy()
            best_r = r.copy()
            
    # Phase 1: LP-driven Simulated Annealing on centers
    curr_c = best_c.copy()
    curr_sum = best_sum
    temp = 0.04
    step = 0.035
    
    for it in range(8000):
        move_type = rng.choice(['move', 'jump'])
        if move_type == 'move':
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            curr_c[idx] += rng.normal(0, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
        else:
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            curr_c[idx] = rng.uniform(0.05, 0.95, 2)
            
        r_new, s_new = solve_lp_radii(curr_c)
        delta = s_new - curr_sum
        
        if delta > 0 or (temp > 1e-8 and rng.random() < np.exp(min(0.0, delta / temp))):
            curr_sum = s_new
            if s_new > best_sum:
                best_sum = s_new
                best_c = curr_c.copy()
                best_r = r_new.copy()
                step = min(0.05, step * 1.02)
        else:
            curr_c[idx] = old
            step = max(0.001, step * 0.995)
            
        temp *= 0.9993
        
    # Phase 2: SLSQP refinement with boundary-safe parameterization
    bounds_slqp = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_slqp}
    
    p0 = to_params(best_c, np.clip(best_r * 0.995, 1e-6, 0.49))
    try:
        res = minimize(obj_slqp, p0, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                       options={'maxiter': 6000, 'ftol': 1e-13})
        if res.success and np.min(cons_slqp(res.x)) >= -1e-7:
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c, best_r = from_params(res.x)
    except Exception:
        pass
        
    # Phase 3: Perturbation & SLSQP to escape remaining local minima
    for k in range(30):
        xp = to_params(best_c, best_r * 0.995)
        xp[:N] += rng.normal(0, 0.002, N)
        xp[N:3*N] += rng.normal(0, 0.025, 2*N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slqp, xp, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-13})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-7:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_c, best_r = from_params(res.x)
        except Exception:
            continue
            
    # Phase 4: High-precision final polish
    try:
        res_f = minimize(obj_slqp, to_params(best_c, best_r * 0.998), method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                         options={'maxiter': 10000, 'ftol': 1e-14})
        if res_f.success and np.min(cons_slqp(res_f.x)) >= -1e-8:
            best_c, best_r = from_params(res_f.x)
            best_sum = -res_f.fun
    except Exception:
        pass
        
    # Final LP to ensure radii are exactly optimal for the final centers
    final_r, final_sum = solve_lp_radii(best_c)
    
    return best_c, np.maximum(final_r, 0.0), float(final_sum)
