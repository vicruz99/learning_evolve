# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000043 (state 8d6d3048) state=347516f5 sum of radii=2.626220 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[:N])

def constraints(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    Boundary constraints are handled automatically by the (r, u, v) parameterization.
    """
    r = vars_vec[:N]
    u = vars_vec[N:2*N]
    v = vars_vec[2*N:3*N]
    
    # Map normalized u, v to actual coordinates within [r, 1-r]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Vectorized pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Only need upper triangular pairs (i < j)
    i_idx, j_idx = np.triu_indices(N, k=1)
    return dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Bounds: r in [1e-6, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -np.inf
    
    inits = []
    
    # 1. Hexagonal lattice initializations
    for seed in range(15):
        np.random.seed(seed * 100)
        pts = []
        ry = 0.09
        y = ry
        row = 0
        while len(pts) < N:
            x_start = ry if row % 2 == 0 else 2.0 * ry
            x = x_start
            while x <= 1.0 - ry and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * ry
            y += np.sqrt(3.0) * ry
            row += 1
        pts = np.array(pts[:N])
        r0 = np.full(N, 0.09)
        u0 = (pts[:, 0] - r0) / (1.0 - 2.0 * r0)
        v0 = (pts[:, 1] - r0) / (1.0 - 2.0 * r0)
        u0 = np.clip(u0 + np.random.uniform(-0.04, 0.04, N), 0.0, 1.0)
        v0 = np.clip(v0 + np.random.uniform(-0.04, 0.04, N), 0.0, 1.0)
        inits.append(np.concatenate([r0, u0, v0]))
        
    # 2. Grid initializations
    for seed in range(15):
        np.random.seed(seed * 100 + 1)
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        r0 = np.full(N, 0.09)
        u0 = (pts[:, 0] - r0) / (1.0 - 2.0 * r0)
        v0 = (pts[:, 1] - r0) / (1.0 - 2.0 * r0)
        u0 = np.clip(u0 + np.random.uniform(-0.04, 0.04, N), 0.0, 1.0)
        v0 = np.clip(v0 + np.random.uniform(-0.04, 0.04, N), 0.0, 1.0)
        inits.append(np.concatenate([r0, u0, v0]))
        
    # 3. Random strictly feasible initializations
    for seed in range(20):
        np.random.seed(seed * 100 + 2)
        pts = np.random.rand(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        wall_dists = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]), np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
        # Set radii to 95% of tightest limit to guarantee strict feasibility
        max_r = np.minimum(wall_dists, min_dists / 2.0) * 0.95
        r0 = max_r
        u0 = (pts[:, 0] - r0) / (1.0 - 2.0 * r0)
        v0 = (pts[:, 1] - r0) / (1.0 - 2.0 * r0)
        u0 = np.clip(u0, 0.0, 1.0)
        v0 = np.clip(v0, 0.0, 1.0)
        inits.append(np.concatenate([r0, u0, v0]))
        
    # Phase 1: Broad search
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
                    s_val = np.sum(res.x[:N])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative refinement around best solution
    if best_vars is not None:
        curr_vars = best_vars.copy()
        for k in range(15):
            pert = curr_vars.copy()
            pert[:N] += np.random.uniform(-0.003, 0.003, N)
            pert[N:2*N] += np.random.uniform(-0.03, 0.03, N)
            pert[2*N:3*N] += np.random.uniform(-0.03, 0.03, N)
            pert[:N] = np.clip(pert[:N], 1e-6, 0.5)
            pert[N:2*N] = np.clip(pert[N:2*N], 0.0, 1.0)
            pert[2*N:3*N] = np.clip(pert[2*N:3*N], 0.0, 1.0)
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
                if res.success:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-6:
                        s_val = np.sum(res.x[:N])
                        if s_val > best_sum:
                            best_sum = s_val
                            best_vars = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-14})
            if res.success and np.min(constraints(res.x)) >= -1e-6:
                best_vars = res.x
        except Exception:
            pass

    # Fallback (should not be reached)
    if best_vars is None:
        r_f = np.full(N, 0.05)
        u_f = np.linspace(0, 1, N)
        v_f = np.linspace(0, 1, N)
        best_vars = np.concatenate([r_f, u_f, v_f])
        
    # Reconstruct centers and radii
    r_opt = best_vars[:N]
    u_opt = best_vars[N:2*N]
    v_opt = best_vars[2*N:3*N]
    
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(np.sum(r_opt))
