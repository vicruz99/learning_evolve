# sol_000169 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000057 (state 347516f5) state=ec6a54df sum of radii=2.622496 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_con = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_con, n))
    b_ub = np.zeros(num_con)
    k = 0
    
    for i in range(n):
        x, y = centers[i]
        for b in (x, 1.0 - x, y, 1.0 - y):
            A_ub[k, i] = 1.0
            b_ub[k] = b
            k += 1
            
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
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 1e-6

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

def obj_slqp(p):
    """Objective: minimize negative sum of radii."""
    return -np.sum(p[:N])

def cons_slqp(p):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = p[:N]
    u = p[N:2*N]
    v = p[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    
    # 1. Hexagonal lattice patterns with varied row counts
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [7,5,5,5,4], 
        [5,5,5,5,6], [6,6,5,5,4], [5,5,6,5,5], [6,4,6,5,5]
    ]
    
    for pat in patterns:
        for _ in range(3):
            pts = []
            r_est = 0.095
            y = r_est
            row = 0
            for cnt in pat:
                shift = 0.0 if row % 2 == 0 else r_est
                x = r_est + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += 2.0 * r_est
                y += np.sqrt(3.0) * r_est
                row += 1
                
            pts = np.array(pts[:N])
            pts = (pts - 0.5) * rng.uniform(0.88, 1.12) + 0.5
            
            rot = rng.uniform(-0.25, 0.25)
            c_val, s_val = np.cos(rot), np.sin(rot)
            pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
            
            pts += rng.uniform(-0.015, 0.015, pts.shape)
            inits.append(np.clip(pts, 0.02, 0.98))
            
    # 2. Force-directed layouts for organic packings
    for s in range(12):
        rng_fd = np.random.RandomState(s)
        pts = rng_fd.rand(N, 2) * 0.8 + 0.1
        for _ in range(250):
            f = np.zeros_like(pts)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff**2, axis=2)) + 1e-4
            f += np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
            pts += 0.0025 * f
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.seterr(all='ignore')
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.49)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_slqp}
    
    best_p = None
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    inits = generate_inits(rng)
    
    # Evaluate all inits with LP to find the most promising starting topologies
    for c0 in inits:
        r0, s0 = solve_lp(c0)
        if s0 > best_sum:
            best_sum = s0
            best_c = c0.copy()
            best_r = r0.copy()
            
    # Phase 1: LP-Driven Simulated Annealing on Centers to explore topology
    curr_c = best_c.copy()
    curr_sum = best_sum
    temp = 0.045
    step = 0.035
    
    for it in range(5000):
        op = rng.choice(['move', 'swap'])
        if op == 'move':
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            curr_c[idx] += rng.normal(0, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
        else:
            i, j = rng.choice(N, 2, replace=False)
            old = curr_c[[i, j]].copy()
            curr_c[i], curr_c[j] = curr_c[j].copy(), curr_c[i].copy()
            
        nr, ns = solve_lp(curr_c)
        delta = ns - curr_sum
        
        if delta > 0 or (temp > 1e-8 and rng.random() < np.exp(delta / temp)):
            curr_sum = ns
            if ns > best_sum:
                best_sum = ns
                best_c = curr_c.copy()
                best_r = nr.copy()
        else:
            if op == 'move':
                curr_c[idx] = old
            else:
                curr_c[[i, j]] = old
                
        temp *= 0.9993
        step = max(0.001, step * 0.9994)
        
    # Phase 2: SLSQP refinement with boundary-safe parameterization
    r_init = np.clip(best_r * 0.995, 1e-6, 0.49)
    best_p = to_params(best_c, r_init)
    
    for k in range(35):
        xp = best_p.copy()
        xp[:N] += rng.normal(0, 0.002, N)
        xp[N:3*N] += rng.normal(0, 0.025, 2*N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slqp, xp, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-13})
            if res.success and np.min(cons_slqp(res.x)) >= -1e-7:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_p = res.x.copy()
        except Exception:
            continue
            
    # Phase 3: Global rotation escape & re-optimization
    if best_c is not None:
        for _ in range(15):
            rot = rng.uniform(-0.12, 0.12)
            c_val, s_val = np.cos(rot), np.sin(rot)
            mat = np.array([[c_val, -s_val], [s_val, c_val]])
            c_rot = (best_c - 0.5) @ mat.T + 0.5
            c_rot = np.clip(c_rot, 0.01, 0.99)
            
            r_rot, _ = solve_lp(c_rot)
            p_rot = to_params(c_rot, np.clip(r_rot * 0.995, 1e-6, 0.49))
            try:
                res_r = minimize(obj_slqp, p_rot, method='SLSQP', bounds=bounds_slqp,
                                 constraints=cons_dict, options={'maxiter': 4000, 'ftol': 1e-13})
                if res_r.success and np.min(cons_slqp(res_r.x)) >= -1e-7:
                    s_r = -res_r.fun
                    if s_r > best_sum:
                        best_sum = s_r
                        best_p = res_r.x.copy()
                        best_c, best_r = from_params(best_p)
            except Exception:
                pass
                
    # Phase 4: High-precision final polish
    try:
        res_f = minimize(obj_slqp, best_p, method='SLSQP', bounds=bounds_slqp,
                         constraints=cons_dict, options={'maxiter': 15000, 'ftol': 1e-14})
        if res_f.success and np.min(cons_slqp(res_f.x)) >= -1e-8:
            best_p = res_f.x
            best_sum = -res_f.fun
    except Exception:
        pass
        
    centers, radii = from_params(best_p)
    # Final LP to ensure radii are exactly optimal for the exact final centers
    radii_final, sum_final = solve_lp(centers)
    
    return centers, np.maximum(radii_final, 0.0), float(sum_final)
