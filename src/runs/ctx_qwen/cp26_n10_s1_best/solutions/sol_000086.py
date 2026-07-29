# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000067 (state 4f336c07) state=91fe82b7 sum of radii=2.623358 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIL_I, TRIL_J = np.tril_indices(N, -1)

def compute_constraints(x):
    """Compute inequality constraints g(x) >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    m = 4 * N + N * (N - 1) // 2
    g = np.empty(m)
    idx = 0
    for i in range(N):
        g[idx] = cx[i] - r[i]
        g[idx+1] = 1.0 - cx[i] - r[i]
        g[idx+2] = cy[i] - r[i]
        g[idx+3] = 1.0 - cy[i] - r[i]
        idx += 4
        
    dx = cx[TRIL_I] - cx[TRIL_J]
    dy = cy[TRIL_I] - cy[TRIL_J]
    dr = r[TRIL_I] + r[TRIL_J]
    g[idx:] = np.hypot(dx, dy) - dr
    return g

def compute_jacobian(x):
    """Compute exact Jacobian matrix of constraints."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    m = 4 * N + N * (N - 1) // 2
    J = np.zeros((m, 3 * N))
    
    idx = 0
    for i in range(N):
        J[idx, 3*i] = 1.0; J[idx, 3*i+2] = -1.0
        J[idx+1, 3*i] = -1.0; J[idx+1, 3*i+2] = -1.0
        J[idx+2, 3*i+1] = 1.0; J[idx+2, 3*i+2] = -1.0
        J[idx+3, 3*i+1] = -1.0; J[idx+3, 3*i+2] = -1.0
        idx += 4
        
    dx = cx[TRIL_I] - cx[TRIL_J]
    dy = cy[TRIL_I] - cy[TRIL_J]
    dist = np.hypot(dx, dy)
    safe_dist = np.maximum(dist, 1e-12)
    d_dx = dx / safe_dist
    d_dy = dy / safe_dist
    
    for k in range(len(TRIL_I)):
        i = TRIL_I[k]
        j = TRIL_J[k]
        row = idx + k
        J[row, 3*i] = d_dx[k]
        J[row, 3*j] = -d_dx[k]
        J[row, 3*i+1] = d_dy[k]
        J[row, 3*j+1] = -d_dy[k]
        J[row, 3*i+2] = -1.0
        J[row, 3*j+2] = -1.0
    return J

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    for i in range(n):
        x_val, y_val = centers[i]
        bound = min(x_val, 1.0 - x_val, y_val, 1.0 - y_val)
        A_ub[idx, i] = 1.0
        b_ub[idx] = bound
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0) * 0.99995
    except Exception:
        pass
    return None

def make_hex_init(r0, angle=0.0, seed=0):
    """Generate a hexagonal lattice initialization with optional rotation."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    radii = np.full(N, r0)
    idx = 0
    y = r0
    row = 0
    while idx < N and y + r0 <= 1.0:
        start_x = r0 if row % 2 == 0 else 2.0 * r0
        cx = start_x
        while idx < N and cx + r0 <= 1.0:
            centers[idx] = [cx, y]
            idx += 1
            cx += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    while idx < N:
        centers[idx] = np.random.uniform(0.1, 0.9, 2)
        idx += 1
        
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        centers = (centers - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    centers += np.random.uniform(-0.002, 0.002, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def project_to_bounds(x, shrink=0.0):
    """Ensure variables respect boundary and radius constraints."""
    r = np.maximum(x[2::3], 0.005)
    x = x.copy()
    x[0::3] = np.clip(x[0::3], r, 1.0 - r)
    x[1::3] = np.clip(x[1::3], r, 1.0 - r)
    x[2::3] = r
    if shrink > 0:
        x[2::3] *= (1.0 - shrink)
        x[0::3] = np.clip(x[0::3], x[2::3], 1.0 - x[2::3])
        x[1::3] = np.clip(x[1::3], x[2::3], 1.0 - x[2::3])
    return x

def objective(x):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(x[2::3])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    bounds_list = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints, 'jac': compute_jacobian}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Multi-start with diverse rotated hexagonal configurations
    inits = []
    for r0 in np.linspace(0.08, 0.105, 6):
        for ang in np.linspace(0.0, 0.5, 6):
            inits.append(make_hex_init(r0, ang))
    for s in range(15):
        inits.append(make_hex_init(0.09, 0.0, seed=s * 137))
        
    for x0 in inits:
        x0 = project_to_bounds(x0)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_list,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                g = compute_constraints(res.x)
                if np.min(g) >= -1e-6 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating LP radii refinement + SLSQP position optimization + perturbation search
    if best_x is not None:
        for step in range(40):
            # LP radii update
            centers_cur = best_x.reshape(N, 3)[:, :2]
            r_lp = solve_lp_radii(centers_cur)
            if r_lp is not None:
                best_x[2::3] = r_lp
                curr_sum = np.sum(r_lp)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    
                # Re-optimize positions with updated radii
                x0 = project_to_bounds(best_x)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_list,
                                   constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                    if not np.isnan(res.fun):
                        g = compute_constraints(res.x)
                        if np.min(g) >= -1e-6 and -res.fun > best_sum:
                            best_sum = -res.fun
                            best_x = res.x.copy()
                except Exception:
                    pass
            
            # Perturbation restart to escape local minima
            x0 = best_x.copy()
            noise = 0.002 * (0.85 ** step)
            x0 += np.random.normal(0, noise, 3 * N)
            x0 = project_to_bounds(x0, shrink=0.01)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds_list,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                if not np.isnan(res.fun):
                    g = compute_constraints(res.x)
                    if np.min(g) >= -1e-6 and -res.fun > best_sum:
                        best_sum = -res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

    # Extract results
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x[2::3]
    
    # Final safety shrink to guarantee strict compliance with validator tolerance
    for _ in range(50):
        g = compute_constraints(best_x)
        if np.min(g) >= -1e-9:
            break
        radii *= 0.999
        for i in range(N):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
        best_x[0::3] = centers[:, 0]
        best_x[1::3] = centers[:, 1]
        best_x[2::3] = radii
        
    return centers, radii, float(np.sum(radii))
