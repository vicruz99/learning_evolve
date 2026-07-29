# sol_000099 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000064 (state 39c4bccd) state=2bfa90af sum of radii=2.631350 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_vec[2::3])

def constraints(vars_vec):
    """
    Inequality constraints: boundary containment and pairwise non-overlap.
    Returns array of values that must be >= 0.
    """
    cx = vars_vec[0::3]
    cy = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def compute_lp_radii(centers):
    """Given fixed centers, solves LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    for i in range(n):
        bounds_val = [x[i], 1.0-x[i], y[i], 1.0-y[i]]
        for b in bounds_val:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i,j)
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
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 1e-5

def generate_hex_init(row_counts, rotation=0.0, scale=1.0, jitter=0.02, seed=0):
    """Generates a hexagonal lattice initialization with specified parameters."""
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
    pts = pts - 0.5
    pts = pts * scale
    pts = pts + 0.5
    
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T
        pts = pts - pts.mean(axis=0) + 0.5
        
    pts = pts + rng.uniform(-jitter, jitter, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -np.inf
    best_x = None
    
    # Diverse row count patterns for hexagonal packing
    patterns = [
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 5, 5], [5, 5, 5, 6, 5], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [5, 6, 4, 6, 5], [4, 6, 5, 6, 5],
        [7, 6, 5, 4, 4], [5, 7, 6, 5, 3], [8, 6, 5, 4, 3], [6, 7, 6, 5, 2],
        [6, 6, 6, 5, 3], [5, 5, 7, 5, 4], [7, 5, 7, 4, 3]
    ]
    
    configs = []
    # Generate diverse configurations
    for pat in patterns:
        for rot in [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
            for sc in [0.85, 0.95, 1.0, 1.05, 1.15]:
                centers = generate_hex_init(pat, rotation=rot, scale=sc, jitter=0.025, seed=len(configs))
                r_init, _ = compute_lp_radii(centers)
                x0 = np.zeros(3*N)
                x0[0::3] = centers[:, 0]
                x0[1::3] = centers[:, 1]
                # Shrink radii slightly to ensure strict feasibility for SLSQP start
                x0[2::3] = np.maximum(r_init, 1e-5) * 0.99
                configs.append(x0)
                
    rng = np.random.RandomState(42)
    rng.shuffle(configs)
    
    # Phase 1: Broad search from diverse initializations
    for x0 in configs[:70]:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is not None:
        # Phase 2: Local perturbation refinement to escape local minima & symmetry
        for k in range(40):
            x_p = best_x + np.random.randn(3*N) * 0.004
            x_p[0::3] = np.clip(x_p[0::3], 0.01, 0.99)
            x_p[1::3] = np.clip(x_p[1::3], 0.01, 0.99)
            x_p[2::3] = np.clip(x_p[2::3], 1e-5, 0.49)
            
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if res.success and np.min(constraints(res.x)) >= -1e-8:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res_f = minimize(objective, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                             options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_f.success and np.min(constraints(res_f.x)) >= -1e-9:
                best_x = res_f.x
                best_sum = -res_f.fun
        except Exception:
            pass
            
    # Fallback configuration if optimization fails completely
    if best_x is None:
        centers = np.array([[0.1+i*0.2, 0.1+j*0.2] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        r_init, _ = compute_lp_radii(centers)
        best_x = np.zeros(3*N)
        best_x[0::3] = centers[:, 0]
        best_x[1::3] = centers[:, 1]
        best_x[2::3] = r_init
        best_sum = np.sum(r_init)
        
    centers_out = np.column_stack((best_x[0::3], best_x[1::3]))
    radii_out = best_x[2::3]
    return centers_out, radii_out, float(best_sum)
