# sol_000134 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000123 (state 6c823a1e) state=7b986201 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_IDX = np.tril_indices(N, -1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0 (vectorized)."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        xs - rs, 1.0 - xs - rs,
        ys - rs, 1.0 - ys - rs
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    c = np.concatenate([c, dx[TRIL_IDX]**2 + dy[TRIL_IDX]**2 - dr[TRIL_IDX]**2])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = max(0.0, dist)
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def project_to_feasible(centers, radii):
    """Ensure centers and radii strictly respect bounds."""
    centers = centers.copy()
    radii = radii.copy()
    for i in range(N):
        r = max(1e-7, radii[i])
        radii[i] = r
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
    return centers, radii

def make_hex_init(r0, angle=0.0, shift=(0.0, 0.0)):
    """Generate hexagonal lattice initialization with rotation and shift."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 10:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x <= 1.0 - r0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        if y > 1.0 + r0: break
        
    pts = np.array(pts[:N + 10])
    pts += shift
    
    if angle != 0.0:
        c_a, s_a = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_a, -s_a], [s_a, c_a]]) + 0.5
        
    mask = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.99) & (pts[:, 1] >= 0.01) & (pts[:, 1] <= 0.99)
    pts = pts[mask]
    
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(0.15, 0.85, (1, 2))])
    return pts[:N]

def force_spread_init(seed, steps=400):
    """Force-directed spread initialization pushing to boundaries."""
    np.random.seed(seed)
    pts = np.random.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.05)
    
    for step in range(steps):
        f = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[j] - pts[i]
                d = np.linalg.norm(diff)
                if d < 0.3 and d > 1e-5:
                    force = 0.002 / (d**2 + 0.001)
                    f[i] -= diff * force / d
                    f[j] += diff * force / d
            for dim in range(2):
                if pts[i, dim] < radii[i] + 0.05: f[i, dim] += 0.06
                elif pts[i, dim] > 1.0 - radii[i] - 0.05: f[i, dim] -= 0.06
        lr = 0.05 * (1.0 - step / steps)
        pts += f * lr
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_sum = -1.0
    best_c = None
    best_r = None
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Phase 1: Diverse Initial Configurations
    inits = []
    for r0 in np.linspace(0.075, 0.105, 6):
        for ang in np.linspace(-0.4, 0.4, 7):
            inits.append(make_hex_init(r0, ang))
    for s in range(8):
        inits.append(force_spread_init(s))
        
    # Phase 2: Multi-start Optimization
    for pts in inits:
        r_lp = solve_lp_radii(pts)
        if r_lp is None: continue
        
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_lp * 0.995
        x0 = np.zeros(3 * N)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_lp * 0.995
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                c_opt = res.x[0::3].reshape(N, 2)
                r_opt = res.x[2::3]
                r_lp_new = solve_lp_radii(c_opt)
                if r_lp_new is not None:
                    r_opt = r_lp_new
                curr_sum = np.sum(r_opt)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            pass

    # Phase 3: LP-Evaluated Hill Climbing with SLSQP Refinement
    if best_c is not None:
        best_c, best_r = project_to_feasible(best_c, best_r)
        best_sum = np.sum(best_r)
        
        for step in range(250):
            # Cooling schedule
            noise_scale = 0.008 * (0.965 ** step)
            
            c_trial = best_c + np.random.normal(0, noise_scale, best_c.shape)
            c_trial = np.clip(c_trial, 0.01, 0.99)
            
            # Occasional swap to break topology symmetry
            if step % 15 == 0:
                i_s, j_s = np.random.choice(N, 2, replace=False)
                c_trial[i_s], c_trial[j_s] = c_trial[j_s].copy(), c_trial[i_s].copy()
                
            r_trial = solve_lp_radii(c_trial)
            if r_trial is None: continue
            
            s_trial = np.sum(r_trial)
            
            if s_trial > best_sum:
                best_c = c_trial.copy()
                best_r = r_trial.copy()
                best_sum = s_trial
                
                # Refine topology with SLSQP
                x0 = np.zeros(3 * N)
                x0[0::3] = best_c[:, 0]
                x0[1::3] = best_c[:, 1]
                x0[2::3] = best_r * 0.985
                x0[0::3] = np.clip(x0[0::3], x0[2::3], 1.0 - x0[2::3])
                x0[1::3] = np.clip(x0[1::3], x0[2::3], 1.0 - x0[2::3])
                
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                    if not np.isnan(res.fun):
                        c_ref = res.x[0::3].reshape(N, 2)
                        r_ref = solve_lp_radii(c_ref)
                        if r_ref is not None:
                            s_ref = np.sum(r_ref)
                            if s_ref > best_sum:
                                best_c = c_ref.copy()
                                best_r = r_ref.copy()
                                best_sum = s_ref
                except Exception:
                    pass

    # Fallback (rare)
    if best_c is None:
        best_c = make_hex_init(0.09, 0.0)
        best_r = solve_lp_radii(best_c)
        if best_r is None: best_r = np.full(N, 0.06)
        best_sum = np.sum(best_r)
        
    # Final strict validation repair
    centers, radii = project_to_feasible(best_c, best_r)
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or \
               centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False; break
                if not valid: break
        if valid: break
        radii *= 0.9995
        centers, radii = project_to_feasible(centers, radii)
        
    return centers, radii, float(np.sum(radii))
