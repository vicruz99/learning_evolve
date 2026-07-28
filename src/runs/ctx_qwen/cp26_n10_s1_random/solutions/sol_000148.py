# sol_000148 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=41e5ee41 sum of radii=2.627329 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_constraints(vars_array):
    """Computes pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2"""
    r = vars_array[:N]
    u = vars_array[N:2*N]
    v = vars_array[2*N:3*N]
    
    # Parameterization automatically satisfies boundary constraints:
    # x = r + (1-2r)*u  =>  r <= x <= 1-r when u in [0,1]
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    return d2[i_idx, j_idx] - rs[i_idx, j_idx]**2

def objective(vars_array):
    """Objective: minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars_array[:N])

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with various densities
    for r0 in [0.085, 0.09, 0.095, 0.10]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            shift = r0 if row % 2 == 1 else 0.0
            x = r0 + shift
            while x + r0 <= 1.0:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        inits.append(pts)
        
    # 2. Regular grid + center point
    gx = np.linspace(0.12, 0.88, 5)
    gy = np.linspace(0.12, 0.88, 5)
    pts_grid = np.array([(x, y) for y in gy for x in gx])
    pts_grid = np.vstack([pts_grid, [0.5, 0.5]])
    inits.append(pts_grid)
    
    best_sum = -np.inf
    best_vars = None
    
    # Phase 1: Initial optimizations from diverse starts
    for cfg in inits:
        r0 = np.full(N, 0.09)
        denom = 1.0 - 2.0 * r0
        u = np.clip((cfg[:, 0] - r0) / denom, 0.0, 1.0)
        v = np.clip((cfg[:, 1] - r0) / denom, 0.0, 1.0)
        x0 = np.concatenate([r0, u, v])
        
        for _ in range(8):
            xp = x0.copy()
            xp[:N] *= rng.uniform(0.95, 1.05, N)
            xp[N:] += rng.uniform(-0.025, 0.025, 2*N)
            xp = np.clip(xp, [1e-6]*N + [0.0]*(2*N), [0.5]*N + [1.0]*(2*N))
            
            try:
                res = minimize(objective, xp, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                if np.isfinite(res.fun):
                    c_vals = compute_constraints(res.x)
                    if np.min(c_vals) > -1e-4:
                        s = -res.fun
                        if s > best_sum:
                            best_sum = s
                            best_vars = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization unexpectedly fails
    if best_vars is None:
        r0 = np.full(N, 0.09)
        u = np.random.rand(N)
        v = np.random.rand(N)
        best_vars = np.concatenate([r0, u, v])
        best_sum = np.sum(r0)

    # Phase 2: Iterative refinement of the best solution found
    for _ in range(30):
        xp = best_vars.copy()
        xp[:N] *= rng.uniform(0.97, 1.03, N)
        xp[N:] += rng.uniform(-0.015, 0.015, 2*N)
        xp = np.clip(xp, [1e-6]*N + [0.0]*(2*N), [0.5]*N + [1.0]*(2*N))
        
        try:
            res = minimize(objective, xp, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_vals = compute_constraints(res.x)
                if np.min(c_vals) > -1e-4:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    # Decode parameters to final centers
    r_slsqp = best_vars[:N]
    u_out = best_vars[N:2*N]
    v_out = best_vars[2*N:3*N]
    
    x_out = r_slsqp + (1.0 - 2.0 * r_slsqp) * u_out
    y_out = r_slsqp + (1.0 - 2.0 * r_slsqp) * v_out
    centers_out = np.column_stack((x_out, y_out))
    
    # Phase 3: LP Refinement for radii given fixed optimal centers
    # This extracts exact slack from the geometric configuration
    c_obj = -np.ones(N)
    A_ub = []
    b_ub = []
    lp_bounds = []
    
    for i in range(N):
        max_r = min(x_out[i], 1.0 - x_out[i], y_out[i], 1.0 - y_out[i])
        lp_bounds.append((0.0, max(max_r - 1e-9, 1e-9)))
        
        # Boundary constraints: r_i <= distance to each wall
        limits = (x_out[i], 1.0 - x_out[i], y_out[i], 1.0 - y_out[i])
        for lim in limits:
            row = np.zeros(N)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x_out[i] - x_out[j], y_out[i] - y_out[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        lp_res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=lp_bounds, method='highs')
        if lp_res.success:
            r_out = lp_res.x
        else:
            r_out = r_slsqp.copy()
    except Exception:
        r_out = r_slsqp.copy()
        
    # Strict safety scaling to guarantee numerical validity against checker tolerance
    r_out *= 0.999999
    centers_out = np.column_stack((x_out, y_out))
    
    return centers_out, r_out, float(np.sum(r_out))
