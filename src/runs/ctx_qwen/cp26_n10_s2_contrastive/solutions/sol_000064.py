# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000038 (state cf517c54) state=02687ba0 sum of radii=2.604816 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def hex_init(scale=1.0, margin=0.05):
    """Generate a hexagonal lattice initialization scaled to fit in the unit square."""
    pts = []
    r0 = 0.09 * scale
    s = 2.0 * r0
    y = margin + r0
    row = 0
    counts = [6, 5, 6, 5, 4]  # 26 circles total
    for cnt in counts:
        x_start = margin + r0 + (row % 2) * s / 2.0
        for k in range(cnt):
            x = x_start + k * s
            if x <= 1.0 - margin - r0:
                pts.append([x, y])
        y += s * np.sqrt(3) / 2.0
        row += 1
    return np.array(pts[:N])

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to non-overlap and boundaries."""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds_r.append((0.0, max(0.0, mx)))
        
    A_ub = np.zeros((n * (n - 1) // 2, n))
    b_ub = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def s_obj(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * N:])

def s_constr(x):
    """Constraints: boundary containment and pairwise non-overlap."""
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    cons = []
    cons.extend(c[:, 0] - r)
    cons.extend(1.0 - c[:, 0] - r)
    cons.extend(c[:, 1] - r)
    cons.extend(1.0 - c[:, 1] - r)
    
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dists = np.hypot(dx, dy)
    r_sum = r[:, None] + r[None, :]
    idx = np.triu_indices(N, k=1)
    cons.extend(dists[idx] - r_sum[idx])
    return np.array(cons)

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': s_constr}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.RandomState(123)
    
    # Generate diverse initial configurations
    inits = []
    for scale in np.linspace(0.85, 1.15, 15):
        p = hex_init(scale=scale)
        p += rng.normal(0, 0.012, p.shape)
        p = np.clip(p, 0.05, 0.95)
        r0 = 0.065 + rng.uniform(0, 0.01)
        inits.append(np.concatenate([p.flatten(), np.full(N, r0)]))
        
    for _ in range(10):
        p = rng.uniform(0.15, 0.85, (N, 2))
        r0 = 0.06
        inits.append(np.concatenate([p.flatten(), np.full(N, r0)]))
        
    # Stage 1: SLSQP optimization from multiple starts + LP refinement
    for x0 in inits:
        try:
            res = minimize(s_obj, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_opt = res.x[:2 * N].reshape(N, 2)
                r_lp, s_lp = solve_radii_lp(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Stage 2: Local perturbation search around the best found configuration
    if best_centers is not None:
        for _ in range(15):
            pert_c = best_centers + rng.normal(0, 0.004, best_centers.shape)
            pert_c = np.clip(pert_c, 0.05, 0.95)
            pert_r = best_radii + rng.normal(0, 0.002, N)
            pert_r = np.maximum(pert_r, 0.01)
            x0_p = np.concatenate([pert_c.flatten(), pert_r])
            
            try:
                res = minimize(s_obj, x0_p, method='SLSQP', bounds=bounds, constraints=cons_dict,
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_p = res.x[:2 * N].reshape(N, 2)
                    r_lp, s_lp = solve_radii_lp(c_p)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_p.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass
                
    # Fallback if all optimizations fail
    if best_centers is None:
        best_centers = hex_init()
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Stage 3: Deterministic post-processing to guarantee strict validity
    for _ in range(50):
        changed = False
        for i in range(N):
            mx = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mx + 1e-10:
                radii[i] = max(0.0, mx - 1e-10)
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess / 2.0
                    radii[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
