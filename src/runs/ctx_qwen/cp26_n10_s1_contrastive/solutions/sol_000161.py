# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000090 (state e01611c4) state=d192bbf0 sum of radii=2.571664 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_c = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_c, n))
    b_ub = np.zeros(num_c)
    k = 0
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
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
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 0.0

def to_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def from_params(params):
    """Reconstruct physical centers and radii from parameters."""
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
        c_val, s_val = np.cos(rot), np.sin(rot)
        M = np.array([[c_val, -s_val], [s_val, c_val]])
        pts = (pts - 0.5) @ M.T + 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(rng):
    """Generates a force-directed layout initialization."""
    pts = rng.uniform(0.15, 0.85, (N, 2))
    for _ in range(250):
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
    pts += rng.uniform(-0.025, 0.025, pts.shape)
    return np.clip(pts, 0.05, 0.95)

def run_packing():
    rng = np.random.default_rng(42)
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    inits = []
    pats = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [7,5,5,5,4], [5,5,5,5,6], 
        [6,6,5,5,4], [5,5,6,5,5], [6,5,5,5,5], [5,5,5,6,5], [6,4,6,5,5],
        [5,5,5,5,5,1], [4,5,5,5,7], [5,7,5,5,4], [8,5,5,5,3], [4,5,6,5,6]
    ]
    for pat in pats:
        for _ in range(3):
            inits.append(hex_init(rng, pat, rot=rng.uniform(-0.25, 0.25), scale=rng.uniform(0.85, 1.15)))
    for _ in range(12):
        inits.append(force_init(rng))
    for _ in range(6):
        inits.append(grid_init(rng))
        
    # Evaluate inits with LP to find strongest starting topology
    init_scores = []
    for c in inits:
        r, s_val = solve_lp_radii(c)
        init_scores.append((s_val, c, r))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    if init_scores:
        best_sum = init_scores[0][0]
        best_c = init_scores[0][1].copy()
        best_r = init_scores[0][2].copy()
    else:
        best_c = grid_init(rng)
        best_r, best_sum = solve_lp_radii(best_c)
        
    # Phase 1: Simulated Annealing on centers to explore topology
    curr_c = best_c.copy()
    curr_sum = best_sum
    temp = 0.045
    step = 0.035
    
    for it in range(9000):
        move_type = rng.integers(0, 3)
        if move_type == 0:
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            curr_c[idx] += rng.normal(0, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.005, 0.995)
        elif move_type == 1:
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            curr_c[idx] = rng.uniform(0.1, 0.9, 2)
        else:
            i, j = rng.choice(N, 2, replace=False)
            old = curr_c[[i, j]].copy()
            curr_c[i], curr_c[j] = curr_c[j].copy(), curr_c[i].copy()
            
        nr, ns = solve_lp_radii(curr_c)
        delta = ns - curr_sum
        
        if delta > 0 or (temp > 1e-8 and rng.random() < np.exp(min(0.0, delta / temp))):
            curr_sum = ns
            if ns > best_sum:
                best_sum = ns
                best_c = curr_c.copy()
                best_r = nr.copy()
        else:
            if move_type < 2:
                curr_c[idx] = old
            else:
                curr_c[[i, j]] = old
                
        temp *= 0.9993
        step = max(0.001, step * 0.9994)
        
    # Phase 2: SLSQP Refinement with boundary-safe parameterization
    bounds_slqp = [(1e-6, 0.49)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_slqp}
    
    p0 = to_params(best_c, np.clip(best_r * 0.995, 1e-6, 0.49))
    
    try:
        res = minimize(obj_slqp, p0, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                       options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
        if res.success and np.min(cons_slqp(res.x)) >= -1e-8:
            s_val = -res.fun
            if s_val > best_sum:
                best_sum = s_val
                best_c, best_r = from_params(res.x)
    except Exception:
        pass
        
    # Phase 3: Perturbation & Restart SLSQP to escape local minima
    for k in range(25):
        xp = to_params(best_c, best_r * 0.99)
        xp[N:3*N] += rng.uniform(-0.025, 0.025, 2*N)
        xp[:N] += rng.uniform(-0.002, 0.002, N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slqp, xp, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 4500, 'ftol': 1e-12, 'disp': False})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-8:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_c, best_r = from_params(res.x)
        except Exception:
            continue
            
    # Phase 4: High-precision final polish
    if best_c is not None:
        p_final = to_params(best_c, np.clip(best_r * 0.998, 1e-6, 0.49))
        try:
            res_f = minimize(obj_slqp, p_final, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                             options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            if res_f.success and np.min(cons_slqp(res_f.x)) >= -1e-9:
                best_c, best_r = from_params(res_f.x)
        except Exception:
            pass
            
    # Final LP to ensure radii are exactly optimal for the final centers
    final_r, final_sum = solve_lp_radii(best_c)
    best_r = final_r
    best_sum = final_sum
    
    # Safety clipping to guarantee strict feasibility
    best_c = np.clip(best_c, 0.0, 1.0)
    best_r = np.maximum(best_r, 0.0)
    
    return best_c, best_r, float(best_sum)
