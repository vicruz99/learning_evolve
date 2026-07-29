# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000156 (state 1f0b18b2) state=b7aa6bfc sum of radii=2.631094 correctness=1.0
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
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        for b in (x, 1.0 - x, y, 1.0 - y):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.ones(n) * 1e-6, 1e-4

def obj_slqp(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def cons_slqp(params):
    """Inequality constraints: dist^2 >= (r_i + r_j)^2. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def to_params(centers, radii):
    """Convert physical centers/radii to (r, u, v) optimization parameters."""
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

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    
    # Hexagonal lattice patterns with varied row counts
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [6,4,6,5,5], [7,5,5,5,4], [5,7,5,5,4], [6,6,5,5,4]
    ]
    
    for pat in patterns:
        for s in range(3):
            pts = []
            r_est = 0.095
            y = r_est
            row = 0
            for cnt in pat:
                shift = (row % 2) * r_est
                x = r_est + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += 2.0 * r_est
                y += np.sqrt(3.0) * r_est
                row += 1
                
            pts = np.array(pts[:N])
            pts = (pts - 0.5) * rng.uniform(0.9, 1.1) + 0.5
            
            rot = rng.uniform(-0.3, 0.3)
            c_val, s_val = np.cos(rot), np.sin(rot)
            pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
            
            pts += rng.uniform(-0.02, 0.02, pts.shape)
            inits.append(np.clip(pts, 0.02, 0.98))
            
    # Force-directed layouts for organic packings
    for s in range(15):
        rng_fd = np.random.RandomState(s)
        pts = rng_fd.rand(N, 2) * 0.8 + 0.1
        for _ in range(200):
            f = np.zeros_like(pts)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff**2, axis=2)) + 1e-4
            f += np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
            pts += 0.003 * f
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    return inits

def run_packing():
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.49)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_slqp}
    
    best_p = None
    best_sum = -np.inf
    
    inits = generate_inits(rng)
    
    # Evaluate all inits with LP to find the most promising starting topologies
    init_scores = []
    for c0 in inits:
        r0, s0 = solve_lp_radii(c0)
        init_scores.append((s0, c0, r0))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 1: SLSQP refinement on top initial configurations
    top_starts = init_scores[:10]
    for s0, c0, r0 in top_starts:
        p0 = to_params(c0, np.clip(r0 * 0.995, 1e-6, 0.49))
        try:
            res = minimize(obj_slqp, p0, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-8:
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_p = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization failed completely
    if best_p is None:
        best_p = to_params(init_scores[0][1], init_scores[0][2] * 0.99)
        best_sum = -obj_slqp(best_p)
        
    # Phase 2: LP-Driven Simulated Annealing on Centers to explore topology
    curr_c, curr_r = from_params(best_p)
    curr_s = best_sum
    temp = 0.045
    step = 0.035
    
    for it in range(5000):
        idx = rng.integers(N)
        old = curr_c[idx].copy()
        curr_c[idx] += rng.normal(0, step, 2)
        curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
        
        nr, ns = solve_lp_radii(curr_c)
        delta = ns - curr_s
        
        if delta > 0 or (temp > 1e-8 and rng.random() < np.exp(delta / temp)):
            curr_s = ns
            if ns > best_sum:
                best_sum = ns
                best_p = to_params(curr_c, np.clip(nr * 0.995, 1e-6, 0.49))
        else:
            curr_c[idx] = old
            
        temp *= 0.9992
        step = max(0.001, step * 0.9994)
        
    # Phase 3: Perturbation & SLSQP restarts to escape local minima
    for k in range(35):
        xp = best_p.copy()
        xp[:N] += rng.normal(0, 0.003, N)
        xp[N:3*N] += rng.normal(0, 0.02, 2*N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slqp, xp, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-8:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_p = res.x.copy()
        except Exception:
            continue
            
    # Phase 4: High-precision final polish
    try:
        res_f = minimize(obj_slqp, best_p, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                         options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if res_f.success and np.min(cons_slqp(res_f.x)) >= -1e-9:
            best_p = res_f.x
            best_sum = -res_f.fun
    except Exception:
        pass
        
    centers, radii = from_params(best_p)
    return centers, np.maximum(radii, 0.0), float(best_sum)
