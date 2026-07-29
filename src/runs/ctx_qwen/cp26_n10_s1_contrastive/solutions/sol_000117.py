# sol_000117 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000105 (state 007a7b0d) state=35bcb8a1 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(params):
    """Minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraints(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    w = params[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + w * (1.0 - 2.0 * r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx*dx + dy*dy
    rs = r[:, None] + r[None, :]
    
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def solve_lp(centers):
    """Given fixed centers, solves LP to find optimal feasible radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0-x, y, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i,j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)]*n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 0.0

def to_params(centers, radii):
    """Map physical centers/radii to (r, u, w) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    w = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, w])

def hex_start(rng, rot, scale):
    """Generate a hexagonal lattice initialization."""
    pts = []
    r_e = 0.095
    y = r_e
    row = 0
    while len(pts) < N:
        shift = (row % 2) * r_e
        x = r_e + shift
        while x <= 1.0 - r_e and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_e
        y += np.sqrt(3.0) * r_e
        row += 1
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    rad = np.full(N, 0.08)
    return pts, rad

def force_start(rng):
    """Spread points using force-directed layout."""
    pts = rng.uniform(0.1, 0.9, (N, 2))
    rad = np.full(N, 0.08)
    for _ in range(200):
        forces = np.zeros_like(pts)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = np.maximum(dists, 1e-5)
        rep = 1.0 / (dists**2)
        np.fill_diagonal(rep, 0.0)
        forces += np.sum(rep[:, :, None] * diff / dists[:, :, None], axis=1)
        for d in range(2):
            forces[:, d] += 5.0 * np.maximum(0, 0.05 - pts[:, d])
            forces[:, d] -= 5.0 * np.maximum(0, pts[:, d] - 0.95)
        pts += 0.005 * forces
        pts = np.clip(pts, 0.05, 0.95)
    for i in range(N):
        dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        dp = np.min(np.linalg.norm(pts[i] - pts, axis=1))
        rad[i] = 0.8 * min(dw, dp/2.0)
    return pts, rad

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(1e-5, 0.49)]*N + [(0.0, 1.0)]*N + [(0.0, 1.0)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_p = None
    best_s = -np.inf
    
    # Generate diverse initial configurations
    starts = []
    for _ in range(30):
        p, r = hex_start(rng, rng.uniform(-0.3, 0.3), rng.uniform(0.85, 1.15))
        starts.append(to_params(p, r))
    for _ in range(20):
        p, r = force_start(rng)
        starts.append(to_params(p, r))
        
    # Phase 1: Broad SLSQP search
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if res.success and np.min(constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_s:
                    best_s = s
                    best_p = res.x.copy()
        except Exception:
            continue
            
    if best_p is not None:
        # Phase 2: Perturbation refinement
        for _ in range(40):
            xp = best_p + rng.normal(0, 0.004, 3*N)
            xp[:N] = np.clip(xp[:N], 1e-5, 0.49)
            xp[N:] = np.clip(xp[N:], 0.0, 1.0)
            try:
                res = minimize(objective, xp, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res.success and np.min(constraints(res.x)) >= -1e-7:
                    s = -res.fun
                    if s > best_s:
                        best_s = s
                        best_p = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: LP-Driven Local Search on Centers
        r_best = best_p[:N]
        u_best = best_p[N:2*N]
        w_best = best_p[2*N:3*N]
        c_best = np.column_stack((r_best + u_best*(1-2*r_best), r_best + w_best*(1-2*r_best)))
        
        rad_init, lp_sum = solve_lp(c_best)
        if lp_sum > best_s:
            best_s = lp_sum
            best_p = to_params(c_best, rad_init)
            
        # Coordinate ascent: move circles one by one, evaluate LP sum
        current_c = c_best.copy()
        current_s = best_s
        step = 0.03
        for _ in range(300):
            idx = rng.integers(N)
            direction = rng.standard_normal(2)
            direction /= np.linalg.norm(direction)
            nc = current_c.copy()
            nc[idx] = np.clip(nc[idx] + step * direction, 0.02, 0.98)
            _, ns = solve_lp(nc)
            if ns > current_s + 1e-6:
                current_c = nc
                current_s = ns
                step = np.maximum(0.001, step * 0.9)
            else:
                step = np.minimum(0.05, step * 1.02)
                
        if current_s > best_s:
            best_s = current_s
            c_best = current_c
            rad_lp, _ = solve_lp(c_best)
            best_p = to_params(c_best, rad_lp)
            
        # Phase 4: High-precision final polish
        final_p = best_p.copy()
        final_p[:N] *= 0.9999 # Slight shrink for strict interior feasibility
        try:
            res_f = minimize(objective, final_p, method='SLSQP', bounds=bounds, constraints=cons,
                             options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_f.success and np.min(constraints(res_f.x)) >= -1e-8:
                best_p = res_f.x
                best_s = -res_f.fun
        except Exception:
            pass
            
    # Fallback configuration
    if best_p is None:
        pts = np.array([[0.1+0.2*i, 0.1+0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        rad = np.full(N, 0.09)
        best_p = to_params(pts, rad)
        best_s = np.sum(rad)
        
    # Reconstruct final centers and radii
    r_opt = best_p[:N]
    u_opt = best_p[N:2*N]
    w_opt = best_p[2*N:3*N]
    centers = np.column_stack((r_opt + u_opt*(1-2*r_opt), r_opt + w_opt*(1-2*r_opt)))
    radii = np.maximum(r_opt, 0.0)
    
    return centers, radii, float(best_s)
