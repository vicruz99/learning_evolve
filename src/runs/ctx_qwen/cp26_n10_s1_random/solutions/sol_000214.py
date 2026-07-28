# sol_000214 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000165 (state ab534a56) state=9159af40 sum of radii=2.631350 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_func(vars_arr, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[2 * n:])

def compute_constraints(vars_arr, n):
    """Compute inequality constraints >= 0 for valid packing."""
    cx = vars_arr[:n]
    cy = vars_arr[n:2 * n]
    r = vars_arr[2 * n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c1 = cx - r
    c2 = 1.0 - cx - r
    c3 = cy - r
    c4 = 1.0 - cy - r
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Using squared distances avoids square root derivatives and is smoother
    cx_m = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_m = cy[:, np.newaxis] - cy[np.newaxis, :]
    r_m = r[:, np.newaxis] + r[np.newaxis, :]
    
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    
    idx = np.triu_indices(n, k=1)
    c5 = d2[idx] - rs2[idx]
    
    return np.concatenate([c1, c2, c3, c4, c5])

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
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
        mx = max(1e-9, min(x, 1.0 - x, y, 1.0 - y))
        bounds.append((0.0, mx))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            A_ub[idx, i] = 1.0
            b_ub[idx] = lim
            idx += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
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
    """Generates initial positions on a centered hexagonal lattice."""
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
    # Center the configuration in the unit square
    cx_mean, cy_mean = pts.mean(axis=0)
    pts -= np.array([cx_mean - 0.5, cy_mean - 0.5])
    return np.clip(pts, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    row_counts_options = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 5, 5, 6, 4],
        [5, 6, 6, 4, 5], [5, 5, 5, 5, 6], [7, 6, 6, 7]
    ]
    
    inits = []
    for rc in row_counts_options:
        if sum(rc) < n:
            continue
        base = generate_hex_init(n, rc)
        inits.append(base)
        # Add perturbations to break symmetry
        for _ in range(2):
            pert = base + rng.uniform(-0.02, 0.02, (n, 2))
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # Random starts
    for _ in range(5):
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.001, 0.999)] * (2 * n) + [(0.0001, 0.25)] * n
    
    # Phase 1: Multi-start SLSQP
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.085
        
        try:
            res = minimize(
                objective_func, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': compute_constraints, 'args': (n,)},
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
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
            
    # Phase 2: Stochastic Local Search on Centers
    # Directly optimizes the true objective (max sum of radii via LP)
    if best_centers is not None:
        current_centers = best_centers.copy()
        step_size = 0.008
        
        for iteration in range(3000):
            idx = rng.integers(n)
            old_pos = current_centers[idx].copy()
            
            # Random perturbation
            direction = rng.uniform(-1.0, 1.0, 2)
            direction /= np.linalg.norm(direction) + 1e-12
            new_pos = current_centers[idx] + step_size * direction
            new_pos = np.clip(new_pos, 0.005, 0.995)
            
            current_centers[idx] = new_pos
            r_new = solve_lp_radii(current_centers)
            
            if r_new is not None:
                s_new = np.sum(r_new)
                if s_new > best_sum + 1e-10:
                    best_sum = s_new
                    best_radii = r_new.copy()
                    # Occasionally shrink step size to refine
                    if rng.random() < 0.05:
                        step_size *= 0.99
                else:
                    current_centers[idx] = old_pos
                    # If no improvement, slightly reduce step size to settle
                    if rng.random() < 0.1:
                        step_size *= 0.995
            else:
                current_centers[idx] = old_pos
                
        # Update best_centers to the one that achieved best_sum
        best_centers = current_centers.copy()

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
                
    best_radii *= scale * 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
