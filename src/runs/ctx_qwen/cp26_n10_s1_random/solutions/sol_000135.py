# sol_000135 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=f24ba82c sum of radii=2.624556 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_optimal_radii(centers):
    """Solves an LP to find the maximum sum of radii for fixed centers."""
    n = centers.shape[0]
    # Upper bounds from boundaries
    ub = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    ub = np.minimum(ub, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 0.0)
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    bounds = [(0.0, u) for u in ub]
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    idx = 0
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
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective(vars_arr):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_arr[2::3])

def constraints(vars_arr, n):
    """Compute all inequality constraints >= 0 for valid packing."""
    xs = vars_arr[0::3]
    ys = vars_arr[1::3]
    rs = vars_arr[2::3]
    
    con = []
    # Boundary constraints
    con.extend(xs - rs)
    con.extend(1.0 - xs - rs)
    con.extend(ys - rs)
    con.extend(1.0 - ys - rs)
    
    # Pairwise non-overlap constraints (squared distances)
    xs_m = xs[:, None] - xs[None, :]
    ys_m = ys[:, None] - ys[None, :]
    rs_m = rs[:, None] + rs[None, :]
    
    dist_sq = xs_m**2 + ys_m**2
    r_sum_sq = rs_m**2
    
    idx = np.triu_indices(n, k=1)
    con.extend(dist_sq[idx] - r_sum_sq[idx])
    
    return np.array(con)

def generate_hex_init(row_counts, r0=0.095, n=26):
    """Generate initial positions on a hexagonal lattice with specified row counts."""
    pts = []
    y = r0
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = 0.5 - row_width / 2.0 + shift
        for c in range(cnt):
            x = x_start + c * 2 * r0
            if len(pts) < n:
                pts.append([x, y])
        y += np.sqrt(3) * r0
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse hexagonal row configurations summing to >= 26
    row_configs = [
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4],
        [4, 6, 6, 6, 4], [5, 6, 6, 5, 4], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 1]
    ]
    
    inits = []
    rng = np.random.default_rng(42)
    for rc in row_configs:
        inits.append(generate_hex_init(rc, r0=0.095, n=n))
        
    # Add controlled perturbations to explore basins
    for _ in range(6):
        base = inits[0].copy()
        base += rng.uniform(-0.025, 0.025, base.shape)
        inits.append(np.clip(base, 0.05, 0.95))
        
    # Phase 1: Optimize from diverse starts
    for cfg in inits:
        # Compute feasible initial radii using LP
        init_radii, _ = get_optimal_radii(cfg)
        if np.sum(init_radii) == 0:
            init_radii = np.full(n, 0.05)
            
        v0 = np.zeros(3 * n)
        v0[0::3] = cfg[:, 0]
        v0[1::3] = cfg[:, 1]
        v0[2::3] = init_radii
        
        try:
            res = minimize(
                objective, v0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False}
            )
            opt_centers = np.column_stack((res.x[0::3], res.x[1::3]))
            
            # Post-process: exact maximal radii for these centers
            radii_lp, sum_lp = get_optimal_radii(opt_centers)
            
            if sum_lp > best_sum:
                best_sum = sum_lp
                best_centers = opt_centers.copy()
                best_radii = radii_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation search around the best found solution
    if best_centers is not None:
        for _ in range(12):
            pert = best_centers + rng.normal(0, 0.004, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            
            init_radii, _ = get_optimal_radii(pert)
            if np.sum(init_radii) == 0:
                init_radii = np.full(n, 0.05)
                
            v0 = np.zeros(3 * n)
            v0[0::3] = pert[:, 0]
            v0[1::3] = pert[:, 1]
            v0[2::3] = init_radii
            
            try:
                res = minimize(
                    objective, v0, method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                    options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False}
                )
                opt_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                radii_lp, sum_lp = get_optimal_radii(opt_centers)
                
                if sum_lp > best_sum:
                    best_sum = sum_lp
                    best_centers = opt_centers.copy()
                    best_radii = radii_lp.copy()
            except Exception:
                continue

    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = get_optimal_radii(best_centers)
        
    # Final safety scaling to guarantee strict validity within numerical tolerance
    if best_radii is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                rs = best_radii[i] + best_radii[j]
                if rs > 1e-12:
                    scale = min(scale, d/rs)
        best_radii *= scale * 0.999999
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
