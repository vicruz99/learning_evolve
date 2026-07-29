# sol_000139 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000074 (state ebc36b4a) state=702859f2 sum of radii=2.620609 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    n_wall = 4 * n
    n_pair = n * (n - 1) // 2
    A_ub = np.zeros((n_wall + n_pair, n))
    b_ub = np.zeros(n_wall + n_pair)
    
    k = 0
    for i in range(n):
        x, y = centers[i]
        b_vals = [x, 1.0 - x, y, 1.0 - y]
        for b in b_vals:
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
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return np.full(n, 1e-5), 1e-4

def to_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def from_params(params):
    """Map (r, u, v) parameters back to physical centers/radii."""
    n = params.shape[0] // 3
    r = params[:n]
    u = params[n:2*n]
    v = params[2*n:3*n]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack((x, y)), r

def obj_slqp(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraints_slqp(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    r_sum = r[I_IDX] + r[J_IDX]
    return dx**2 + dy**2 - r_sum**2

def generate_hex(seed, rot=0.0, scale=1.0):
    """Generates a hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.098
    y = r_est
    row = 0
    while len(pts) < N:
        shift = (row % 2) * r_est
        x = r_est + shift
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def run_sa(centers, rng):
    """Simulated Annealing on centers using LP objective."""
    n = centers.shape[0]
    curr_c = centers.copy()
    curr_r, curr_sum = solve_lp_radii(curr_c)
    best_c = curr_c.copy()
    best_r = curr_r.copy()
    best_sum = curr_sum
    
    temp = 0.04
    step = 0.025
    
    for it in range(1200):
        idx = rng.integers(n)
        old = curr_c[idx].copy()
        curr_c[idx] += rng.normal(0, step, 2)
        curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
        
        new_r, new_sum = solve_lp_radii(curr_c)
        delta = new_sum - curr_sum
        
        if delta > 0 or (temp > 1e-6 and rng.random() < np.exp(delta / temp)):
            curr_sum = new_sum
            if new_sum > best_sum:
                best_sum = new_sum
                best_c = curr_c.copy()
                best_r = new_r.copy()
        else:
            curr_c[idx] = old
            
        temp *= 0.9992
        step = max(0.001, step * 0.9995)
        
    return best_c, best_r, best_sum

def run_packing():
    rng = np.random.default_rng(42)
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    inits = []
    for s in range(12):
        rot = rng.uniform(-0.3, 0.3)
        sc = rng.uniform(0.90, 1.10)
        inits.append(generate_hex(s, rot, sc))
        
    for pts in inits:
        r, s_val = solve_lp_radii(pts)
        if s_val > best_sum:
            best_sum = s_val
            best_c = pts.copy()
            best_r = r.copy()
            
    # SA refinement to optimize center topology
    curr_c, curr_r, curr_sum = run_sa(best_c, rng)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_c = curr_c.copy()
        best_r = curr_r.copy()
        
    # SLSQP Polish with boundary-safe parameterization
    bounds_slqp = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons_slqp = {'type': 'ineq', 'fun': constraints_slqp}
    
    # Ensure strict interior feasibility for SLSQP start
    r_init = np.maximum(best_r * 0.99, 1e-5)
    x0 = to_params(best_c, r_init)
    
    try:
        res = minimize(obj_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                       constraints=cons_slqp, options={'maxiter': 10000, 'ftol': 1e-13})
        if res.success and np.min(constraints_slqp(res.x)) >= -1e-8:
            s_val = np.sum(res.x[:N])
            if s_val > best_sum:
                best_sum = s_val
                c_tmp, r_tmp = from_params(res.x)
                best_c = c_tmp
                best_r = r_tmp
    except Exception:
        pass
        
    # Perturbation & SLSQP to escape local minima
    for k in range(25):
        x_p = to_params(best_c, best_r * 0.995)
        x_p[N:3*N] += rng.uniform(-0.015, 0.015, 2*N)
        x_p[:N] += rng.uniform(-0.002, 0.002, N)
        x_p[:N] = np.clip(x_p[:N], 1e-6, 0.5)
        x_p[N:3*N] = np.clip(x_p[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slqp, x_p, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 6000, 'ftol': 1e-12})
            if res.success and np.min(constraints_slqp(res.x)) >= -1e-8:
                s_val = np.sum(res.x[:N])
                if s_val > best_sum:
                    best_sum = s_val
                    c_tmp, r_tmp = from_params(res.x)
                    best_c = c_tmp
                    best_r = r_tmp
        except Exception:
            pass
            
    # Final high-precision polish
    if best_c is not None:
        x0 = to_params(best_c, best_r * 0.999)
        try:
            res = minimize(obj_slqp, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 15000, 'ftol': 1e-14})
            if res.success and np.min(constraints_slqp(res.x)) >= -1e-8:
                best_c, best_r = from_params(res.x)
                best_sum = np.sum(best_r)
        except Exception:
            pass
            
    # Fallback (should not be reached)
    if best_c is None:
        best_c = generate_hex(0)
        best_r, best_sum = solve_lp_radii(best_c)
        
    return best_c, np.maximum(best_r, 0.0), float(best_sum)
