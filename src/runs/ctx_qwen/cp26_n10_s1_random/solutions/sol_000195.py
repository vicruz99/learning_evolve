# sol_000195 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000148 (state 41e5ee41) state=45fcd706 sum of radii=2.618067 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_penalty(v, n, mu):
    """Augmented penalty objective for L-BFGS-B."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    # Boundary violation penalties
    pen = np.sum(np.maximum(0, r - c[:, 0])**2)
    pen += np.sum(np.maximum(0, r - (1.0 - c[:, 0]))**2)
    pen += np.sum(np.maximum(0, r - c[:, 1])**2)
    pen += np.sum(np.maximum(0, r - (1.0 - c[:, 1]))**2)
    
    # Overlap violation penalties
    dx = c[:, 0, None] - c[None, :, 0]
    dy = c[:, 1, None] - c[None, :, 1]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    pen += np.sum(np.maximum(0, r_sum - d)**2)
    
    return -np.sum(r) + mu * pen

def slsqp_obj(v, n):
    """Objective for SLSQP: maximize sum of radii."""
    return -np.sum(v[2*n:])

def slsqp_cons(v, n):
    """Inequality constraints for SLSQP: >= 0."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    # Boundary constraints
    con = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    
    # Pairwise non-overlap constraints
    dx = c[:, 0, None] - c[None, :, 0]
    dy = c[:, 1, None] - c[None, :, 1]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    idx = np.triu_indices(n, k=1)
    con.append(d[idx] - r_sum[idx])
    
    return np.concatenate(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    best_vars = None
    best_sum = -np.inf
    mu_schedule = [10, 50, 250, 1000, 5000]
    
    # Phase 1: Penalty-based force relaxation with multiple starts
    for trial in range(4):
        # Generate hexagonal lattice initialization
        pts = []
        r_init = 0.085 + trial * 0.003
        y = r_init
        row = 0
        while len(pts) < n + 5:
            shift = r_init if row % 2 == 1 else 0.0
            x = r_init + shift
            while x + r_init <= 1.0:
                pts.append([x, y])
                x += 2.0 * r_init
            y += np.sqrt(3) * r_init
            row += 1
        centers = np.array(pts[:n])
        radii = np.full(n, r_init)
        
        # Add controlled noise to break symmetry
        centers += rng.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        current_vars = np.concatenate([centers.flatten(), radii])
        bounds = [(0.0, 1.0)] * (2*n) + [(1e-4, 0.5)] * n
        
        # Continuation method: increase penalty stiffness gradually
        for mu in mu_schedule:
            res = minimize(compute_penalty, current_vars, args=(n, mu), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-13})
            current_vars = res.x
            
        c_opt = current_vars[:2*n].reshape(n, 2)
        r_opt = current_vars[2*n:]
        s = np.sum(r_opt)
        if s > best_sum:
            best_sum = s
            best_vars = current_vars.copy()
            
    # Phase 2: Strict constraint satisfaction via SLSQP
    slsqp_bounds = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
    best_vars_perturbed = best_vars + rng.uniform(-1e-5, 1e-5, best_vars.shape)
    
    res_slsqp = minimize(slsqp_obj, best_vars_perturbed, args=(n,), method='SLSQP', 
                         bounds=slsqp_bounds,
                         constraints={'type': 'ineq', 'fun': slsqp_cons, 'args': (n,)},
                         options={'maxiter': 10000, 'ftol': 1e-13})
                         
    centers = res_slsqp.x[:2*n].reshape(n, 2)
    radii = res_slsqp.x[2*n:]
    
    # Phase 3: LP refinement to extract maximum radii for fixed optimal centers
    c_obj_lp = -np.ones(n)
    A_ub = []
    b_ub = []
    lp_bounds = []
    
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        lp_bounds.append((0.0, max(max_r - 1e-9, 1e-9)))
        limits = (centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for lim in limits:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        lp_res = linprog(c_obj_lp, A_ub=A_ub, b_ub=b_ub, bounds=lp_bounds, method='highs')
        if lp_res.success:
            radii = lp_res.x
    except Exception:
        pass
        
    # Final numerical safety scaling
    radii *= 0.9999995
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    return centers, radii, float(np.sum(radii))
