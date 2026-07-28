# sol_000165 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000160 (state 296f36e1) state=ab534a56 sum of radii=2.633033 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective(vars_arr, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def constraints(vars_arr, n):
    """Compute inequality constraints >= 0 for valid packing."""
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    
    c = []
    # Boundary constraints: circles inside [0,1]x[0,1]
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    
    dists = np.sqrt(cx_m**2 + cy_m**2)
    np.fill_diagonal(dists, 1e9)
    
    idx = np.triu_indices(n, k=1)
    c.extend(dists[idx] - r_m[idx])
    
    return np.array(c)

def solve_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m_bound = 4 * n
    m_pair = n * (n - 1) // 2
    
    A_ub = np.zeros((m_bound + m_pair, n))
    b_ub = np.zeros(m_bound + m_pair)
    bounds = []
    
    idx = 0
    for i in range(n):
        x, y = centers[i]
        mx = max(0.0, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            A_ub[idx, i] = 1.0
            b_ub[idx] = lim
            idx += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def generate_hex_init(n, row_counts):
    """Generates initial positions on a hexagonal lattice."""
    pts = []
    r0 = 0.09
    y = r0
    for i, cnt in enumerate(row_counts):
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        
    while len(pts) < n:
        pts.append([0.5, 0.5])
        
    pts = np.array(pts[:n])
    cx_mean, cy_mean = pts.mean(axis=0)
    pts -= np.array([cx_mean - 0.5, cy_mean - 0.5])
    return np.clip(pts, 0.05, 0.95)

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    row_counts_options = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 5, 5, 6, 4],
        [5, 6, 6, 4, 5], [5, 5, 5, 5, 6], [7, 6, 6, 7],
        [6, 7, 6, 7], [5, 6, 5, 6, 5, 1]
    ]
    
    inits = []
    for rc in row_counts_options:
        if sum(rc) < n:
            continue
        base = generate_hex_init(n, rc)
        inits.append(base)
        # Add perturbations to break symmetry
        for _ in range(3):
            pert = base + rng.uniform(-0.025, 0.025, (n, 2))
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # Random starts
    for _ in range(8):
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.001, 0.999)] * (2 * n) + [(0.0001, 0.25)] * n
    
    # Phase 1: Multi-start constrained optimization
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.08
        
        try:
            res = minimize(
                objective, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
            )
            
            if not np.isfinite(res.fun):
                continue
                
            c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
            r_lp = solve_lp_radii(c_opt)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement around best found configuration
    if best_centers is not None:
        for _ in range(15):
            pert = best_centers + rng.normal(0, 0.004, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            
            x0 = np.zeros(3 * n)
            x0[:n] = pert[:, 0]
            x0[n:2*n] = pert[:, 1]
            x0[2*n:] = best_radii * 0.96
            
            try:
                res = minimize(
                    objective, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                    constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                    options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False}
                )
                
                if not np.isfinite(res.fun):
                    continue
                    
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
            except Exception:
                continue

    # Fallback configuration
    if best_centers is None:
        best_centers = generate_hex_init(n, [6, 5, 6, 5, 4])
        best_radii = np.full(n, 0.085)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
