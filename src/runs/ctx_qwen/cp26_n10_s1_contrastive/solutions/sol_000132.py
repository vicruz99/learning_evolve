# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000123 (state a85a7f81) state=260c4e7f sum of radii=2.619889 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp(centers):
    """Given fixed centers, solves LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    dist_wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                           np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    A_ub_b = np.eye(n)
    b_ub_b = dist_wall
    
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    I, J = np.triu_indices(n, k=1)
    n_pairs = len(I)
    A_ub_p = np.zeros((n_pairs, n))
    A_ub_p[np.arange(n_pairs), I] = 1.0
    A_ub_p[np.arange(n_pairs), J] = 1.0
    b_ub_p = dists[I, J]
    
    A_ub = np.vstack([A_ub_b, A_ub_p])
    b_ub = np.concatenate([b_ub_b, b_ub_p])
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.full(n, 1e-5), 0.0

def run_sa(centers, steps=600, init_temp=0.04):
    """Simulated Annealing on centers, evaluating objective via LP."""
    rng = np.random.default_rng(1234)
    cur_c = centers.copy()
    cur_r, cur_sum = solve_lp(cur_c)
    best_c = cur_c.copy()
    best_r = cur_r.copy()
    best_sum = cur_sum
    
    T = init_temp
    for i in range(steps):
        idx = rng.integers(N)
        old = cur_c[idx].copy()
        step = rng.uniform(-0.04, 0.04, 2)
        cur_c[idx] = np.clip(cur_c[idx] + step, 1e-4, 1.0 - 1e-4)
        
        new_r, new_sum = solve_lp(cur_c)
        delta = new_sum - cur_sum
        
        if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
            cur_sum = new_sum
            if new_sum > best_sum:
                best_sum = new_sum
                best_c = cur_c.copy()
                best_r = new_r.copy()
        else:
            cur_c[idx] = old
            
        T *= 0.995
        
    return best_c, best_r, best_sum

def make_params(centers, radii):
    """Converts physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def objective_slqp(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[:N])

def constraint_slqp(vars):
    """Inequality constraints: pairwise non-overlap using parameterized coordinates."""
    r = vars[:N]
    u = vars[N:2*N]
    v = vars[2*N:3*N]
    
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def generate_hex_init(seed, rot, scale):
    """Generates a hexagonal lattice initialization with rotation and scaling."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.09
    y = r_est
    row = 0
    while len(pts) < N:
        x_off = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_off
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    pts -= 0.5
    pts *= scale
    pts += 0.5
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        pts -= pts.mean(axis=0)
        pts += 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def generate_force_init(seed):
    """Generates an organic layout using repulsive forces."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.1, 0.9, (N, 2))
    for _ in range(200):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-6)
        rep = 1.0 / (dists**2)
        np.fill_diagonal(rep, 0.0)
        forces = np.sum(rep[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
        for d in range(2):
            forces[:, d] += 20.0 * np.maximum(0, 0.05 - pts[:, d])
            forces[:, d] -= 20.0 * np.maximum(0, pts[:, d] - 0.95)
        pts += 0.004 * forces
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds_slqp = [(1e-5, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_slqp = {'type': 'ineq', 'fun': constraint_slqp}
    
    candidates = []
    
    # 1. Hexagonal variations
    for s in range(6):
        rot = np.random.uniform(-0.25, 0.25)
        sc = np.random.uniform(0.90, 1.10)
        candidates.append(generate_hex_init(s, rot, sc))
        
    # 2. Force-directed variations
    for s in range(6):
        candidates.append(generate_force_init(s))
        
    # 3. Grid + Center variations
    for s in range(4):
        rng = np.random.RandomState(s + 100)
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        pts += rng.uniform(-0.02, 0.02, pts.shape)
        candidates.append(np.clip(pts, 0.02, 0.98))
        
    best_c = None
    best_r = None
    best_sum = -np.inf
    
    # Phase 1: Simulated Annealing on centers with LP radius optimization
    for c_init in candidates:
        c_sa, r_sa, s_sa = run_sa(c_init, steps=600, init_temp=0.04)
        if s_sa > best_sum:
            best_sum = s_sa
            best_c = c_sa
            best_r = r_sa
            
    # Phase 2: SLSQP Polish on best SA result
    if best_c is not None:
        r_start = np.maximum(best_r * 0.99, 1e-5)
        x0 = make_params(best_c, r_start)
        
        try:
            res = minimize(objective_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 8000, 'ftol': 1e-13})
            if res.success:
                if np.min(constraint_slqp(res.x)) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        r_opt = res.x[:N]
                        u_opt = res.x[N:2*N]
                        v_opt = res.x[2*N:3*N]
                        best_c = np.column_stack((r_opt + u_opt*(1.0-2.0*r_opt), r_opt + v_opt*(1.0-2.0*r_opt)))
                        best_r = r_opt
        except Exception:
            pass
            
        # Phase 3: Perturbation & SLSQP to escape local minima
        for _ in range(10):
            x0_p = best_c.copy()
            r_p = np.maximum(best_r * 0.99, 1e-5)
            x0_params = make_params(x0_p, r_p)
                
            x0_params[:N] += np.random.uniform(-0.001, 0.001, N)
            x0_params[N:2*N] += np.random.uniform(-0.01, 0.01, N)
            x0_params[2*N:3*N] += np.random.uniform(-0.01, 0.01, N)
            x0_params[:N] = np.clip(x0_params[:N], 1e-5, 0.5)
            x0_params[N:] = np.clip(x0_params[N:], 0.0, 1.0)
            
            try:
                res_p = minimize(objective_slqp, x0_params, method='SLSQP', bounds=bounds_slqp,
                                 constraints=cons_slqp, options={'maxiter': 4000, 'ftol': 1e-12})
                if res_p.success and np.min(constraint_slqp(res_p.x)) >= -1e-7:
                    s_p = -res_p.fun
                    if s_p > best_sum:
                        best_sum = s_p
                        r_opt = res_p.x[:N]
                        u_opt = res_p.x[N:2*N]
                        v_opt = res_p.x[2*N:3*N]
                        best_c = np.column_stack((r_opt + u_opt*(1.0-2.0*r_opt), r_opt + v_opt*(1.0-2.0*r_opt)))
                        best_r = r_opt
            except Exception:
                continue
                
    # Fallback
    if best_c is None:
        best_c = generate_force_init(0)
        best_r, best_sum = solve_lp(best_c)
        
    # Final High-Precision Polish
    try:
        r_start = np.maximum(best_r * 0.999, 1e-5)
        x0_final = make_params(best_c, r_start)
        res_f = minimize(objective_slqp, x0_final, method='SLSQP', bounds=bounds_slqp,
                         constraints=cons_slqp, options={'maxiter': 10000, 'ftol': 1e-14})
        if res_f.success and np.min(constraint_slqp(res_f.x)) >= -1e-8:
            r_opt = res_f.x[:N]
            u_opt = res_f.x[N:2*N]
            v_opt = res_f.x[2*N:3*N]
            best_c = np.column_stack((r_opt + u_opt*(1.0-2.0*r_opt), r_opt + v_opt*(1.0-2.0*r_opt)))
            best_r = r_opt
            best_sum = -res_f.fun
    except Exception:
        pass
        
    return best_c, np.maximum(best_r, 0.0), float(best_sum)
