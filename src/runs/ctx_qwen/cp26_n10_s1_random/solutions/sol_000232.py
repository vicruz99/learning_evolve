# sol_000232 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000142 (state d65765d5) state=47f3a25e sum of radii=2.624551 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(lim, 1e-9)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def hill_climb_lp(centers, rng, steps=300):
    """Local search on centers evaluated via exact LP refinement."""
    n = centers.shape[0]
    best_sum = 0.0
    best_centers = centers.copy()
    radii, current_sum = solve_lp_radii(best_centers)
    best_radii = radii.copy()
    best_sum = current_sum
    
    step_size = 0.02
    
    for _ in range(steps):
        i = rng.integers(n)
        old_pos = best_centers[i].copy()
        
        pert = rng.uniform(-step_size, step_size, 2)
        new_pos = best_centers[i] + pert
        new_pos = np.clip(new_pos, 0.01, 0.99)
        
        best_centers[i] = new_pos
        radii, new_sum = solve_lp_radii(best_centers)
        
        if new_sum > best_sum:
            best_sum = new_sum
            best_radii = radii.copy()
            step_size = max(0.001, step_size * 0.995)
        else:
            best_centers[i] = old_pos
            step_size = min(0.05, step_size * 1.02)
            
    return best_centers, best_radii, best_sum

def slsqp_objective(vars_arr, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(vars_arr[:n])

def slsqp_constraints(vars_arr, n, triu_i, triu_j):
    """Inequality constraints for SLSQP: pairwise non-overlap."""
    r = vars_arr[:n]
    u = vars_arr[n:2*n]
    v = vars_arr[2*n:3*n]
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    d2 = diff_x**2 + diff_y**2
    rs = r[:, None] + r[None, :]
    
    return d2[triu_i, triu_j] - rs[triu_i, triu_j]**2

def joint_optimize(centers_init, n, triu_i, triu_j):
    """Joint optimization of centers and radii using SLSQP."""
    r0 = np.full(n, 0.09)
    u0 = np.clip((centers_init[:, 0] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
    v0 = np.clip((centers_init[:, 1] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
    x0 = np.concatenate([r0, u0, v0])
    
    bounds = [(1e-5, 0.5)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    
    cons = {'type': 'ineq', 'fun': slsqp_constraints, 'args': (n, triu_i, triu_j)}
    
    try:
        res = minimize(slsqp_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14})
        if np.isfinite(res.fun):
            r_opt = res.x[:n]
            u_opt = res.x[n:2*n]
            v_opt = res.x[2*n:3*n]
            denom = 1.0 - 2.0 * r_opt
            x_opt = r_opt + denom * u_opt
            y_opt = r_opt + denom * v_opt
            return np.column_stack((x_opt, y_opt)), r_opt
    except Exception:
        pass
    return centers_init, np.full(n, 0.08)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    triu_i, triu_j = np.triu_indices(n, k=1)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Hexagonal row distributions summing to 26
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [6, 6, 4, 6, 4], [5, 5, 6, 5, 5], [7, 6, 7, 6], [6, 7, 6, 7],
        [6, 5, 4, 6, 5], [5, 6, 6, 5, 4]
    ]
    
    inits = []
    for pat in patterns:
        if sum(pat) != n: 
            continue
        pts = []
        r_init = 0.09
        y = r_init
        for idx, cnt in enumerate(pat):
            shift = r_init if idx % 2 == 1 else 0.0
            x = r_init + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r_init
            y += r_init * np.sqrt(3)
        pts = np.array(pts[:n])
        inits.append(pts)
        
        # Add perturbed variants to break symmetry
        for _ in range(2):
            pert = pts + rng.uniform(-0.02, 0.02, pts.shape)
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # Add fully random starts for diversity
    for _ in range(5):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    for init_c in inits:
        # Phase 1: Joint SLSQP optimization
        c_slsqp, r_slsqp = joint_optimize(init_c, n, triu_i, triu_j)
        
        # Phase 2: Exact LP refinement
        r_lp, s_lp = solve_lp_radii(c_slsqp)
        
        if s_lp > best_sum:
            best_sum = s_lp
            best_centers = c_slsqp.copy()
            best_radii = r_lp.copy()
            
        # Phase 3: Hill climbing on centers with LP evaluation
        c_hp, r_hp, s_hp = hill_climb_lp(c_slsqp, rng, steps=300)
        if s_hp > best_sum:
            best_sum = s_hp
            best_centers = c_hp.copy()
            best_radii = r_hp.copy()
            
    # Final strict safety scaling to guarantee numerical validity
    if best_radii is not None:
        scale = 1.0
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            if r > 1e-9:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                rs = best_radii[i] + best_radii[j]
                if rs > 1e-9:
                    scale = min(scale, d/rs)
        best_radii *= scale * 0.999999
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
