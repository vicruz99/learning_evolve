# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000099 (state 2bfa90af) state=95039334 sum of radii=2.627681 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def obj(params):
    """Objective: maximize sum of radii <=> minimize negative sum."""
    return -np.sum(params[:N])

def constr(params):
    """
    Inequality constraints: pairwise non-overlap.
    Boundary constraints are automatically satisfied by parameterization.
    Returns array of values that must be >= 0.
    """
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for b in [x[i], 1.0-x[i], y[i], 1.0-y[i]]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_r = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-4)

def centers_to_params(centers):
    """Converts physical centers to (r, u, v) optimization parameters."""
    r = solve_lp_radii(centers)
    # Scale down slightly to ensure strict interior feasibility for SLSQP
    r = r * 0.98
    x, y = centers[:, 0], centers[:, 1]
    
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((x - r) / denom, 0.0, 1.0)
    v = np.clip((y - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def hex_init(seed, row_counts, rot=0.0, scale=1.0):
    """Generates a hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    pts = []
    r_est = 0.1
    y = r_est
    for r_idx, count in enumerate(row_counts):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(count):
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = (pts - 0.5) @ R.T + 0.5
        
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def force_init(seed):
    """Generates a force-directed layout initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2) * 0.8 + 0.1
    for _ in range(600):
        f = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist = np.maximum(dist, 1e-4)
        # Coulomb-like repulsion
        f += np.sum((1.0/dist**2)[:, :, None] * diff / dist[:, :, None], axis=1)
        # Wall repulsion
        for d in range(2):
            f[:, d] += 20.0 * np.maximum(0, 0.1 - pts[:, d])
            f[:, d] -= 20.0 * np.maximum(0, pts[:, d] - 0.9)
        pts += 0.004 * f
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def grid_init(seed):
    """Generates a perturbed grid initialization."""
    rng = np.random.RandomState(seed)
    pts = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)])
    pts = np.vstack([pts, [0.5, 0.5]])
    pts += rng.uniform(-0.04, 0.04, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    bounds = [(1e-6, 0.5)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constr}
    
    best_sum = -np.inf
    best_x = None
    
    inits = []
    
    # Diverse hexagonal patterns
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6], [6,4,6,5,5],
        [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3], [5,7,6,5,3], [6,6,6,5,3]
    ]
    idx = 0
    for p in patterns:
        for s in range(4):
            rot = (s % 5 - 2) * 0.08
            sc = 0.9 + (s // 5) * 0.1
            inits.append(hex_init(idx, p, rot=rot, scale=sc))
            idx += 1
            
    # Force layouts
    for s in range(12):
        inits.append(force_init(s))
        
    # Grid layouts
    for s in range(6):
        inits.append(grid_init(s))
        
    # Convert to parameters
    param_inits = []
    for c in inits:
        param_inits.append(centers_to_params(c))
        
    # Phase 1: Broad search
    for x0 in param_inits:
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success:
                c_val = constr(res.x)
                if np.min(c_val) >= -1e-7:
                    s_val = -res.fun
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization failed
    if best_x is None:
        c_f = grid_init(0)
        best_x = centers_to_params(c_f)
        best_sum = -obj(best_x)
        
    # Phase 2: Perturbation refinement to escape local minima
    rng = np.random.RandomState(42)
    for _ in range(35):
        x_p = best_x.copy()
        # Perturb positions more than radii
        x_p[:N] += rng.uniform(-0.002, 0.002, N)
        x_p[N:3*N] += rng.uniform(-0.015, 0.015, 2*N)
        x_p[:N] = np.clip(x_p[:N], 1e-6, 0.49)
        x_p[N:3*N] = np.clip(x_p[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj, x_p, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constr(res.x)) >= -1e-7:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: High-precision polish
    try:
        res_f = minimize(obj, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
        if res_f.success and np.min(constr(res_f.x)) >= -1e-8:
            best_x = res_f.x
            best_sum = -res_f.fun
    except Exception:
        pass
        
    # Reconstruct physical centers
    r = best_x[:N]
    u = best_x[N:2*N]
    v = best_x[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    centers = np.column_stack([x, y])
    radii = r
    
    return centers, radii, float(best_sum)
