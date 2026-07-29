# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 6a99b8b1) state=cd81b075 sum of radii=2.627681 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Returns all inequality constraints >= 0."""
    n = N
    xs, ys, rs = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints
    c_boundary = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    i, j = np.tril_indices(n, -1)
    dx = xs[i] - xs[j]
    dy = ys[i] - ys[j]
    dr = rs[i] + rs[j]
    c_overlap = dx**2 + dy**2 - dr**2
    
    return np.concatenate([c_boundary, c_overlap])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

def make_hex_init(counts, r0):
    """Generates a hexagonal lattice initialization."""
    centers = []
    y = r0
    row = 0
    for cnt in counts:
        x = r0 + (r0 if row % 2 == 1 else 0.0)
        for _ in range(cnt):
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
    while len(centers) < N:
        centers.append([0.5, 0.5])
    centers = np.array(centers[:N])
    # Small perturbation to break symmetry
    centers += np.random.uniform(-0.003, 0.003, centers.shape)
    return centers, np.full(N, r0)

def solve_lp(centers):
    """Optimally scale radii for fixed centers using Linear Programming."""
    n = N
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(lim)
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    try:
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                      bounds=[(0.0, None)]*n, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def run_packing():
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Diverse row patterns summing to 26
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 6, 6, 5, 4],
        [4, 6, 6, 6, 4], [5, 5, 5, 5, 6], [6, 6, 6, 4, 4],
        [7, 6, 6, 5, 2], [6, 6, 6, 6, 2], [5, 7, 6, 5, 3],
        [5, 5, 6, 6, 4], [6, 6, 5, 5, 4], [4, 5, 6, 6, 5]
    ]
    
    inits = []
    for pat in patterns:
        for r in [0.085, 0.090, 0.095, 0.100]:
            c, r_arr = make_hex_init(pat, r)
            inits.append((c, r_arr))
            
    # Phase 1: Multi-start SLSQP
    for c0, r0 in inits:
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0 * 0.5  # Start smaller to guarantee feasibility
        
        # Project to strict feasibility bounds
        for i in range(N):
            x0[3*i] = np.clip(x0[3*i], x0[3*i+2], 1.0-x0[3*i+2])
            x0[3*i+1] = np.clip(x0[3*i+1], x0[3*i+2], 1.0-x0[3*i+2])
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if not np.isnan(res.fun):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    vals = constraints(res.x)
                    if np.min(vals) >= -1e-6:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Alternating LP + SLSQP Refinement
    if best_x is not None:
        for _ in range(10):
            c_best = np.column_stack((best_x[0::3], best_x[1::3]))
            new_r = solve_lp(c_best)
            if new_r is not None:
                best_x[2::3] = new_r
                curr_sum = np.sum(new_r)
                if curr_sum > best_sum:
                    best_sum = curr_sum
                
                # Perturb centers to escape LP plateaus
                x_pert = best_x + np.random.normal(0, 2e-4, 3*N)
                for i in range(N):
                    r = max(0.001, x_pert[3*i+2])
                    x_pert[3*i+2] = r
                    x_pert[3*i] = np.clip(x_pert[3*i], r, 1.0-r)
                    x_pert[3*i+1] = np.clip(x_pert[3*i+1], r, 1.0-r)
                    
                try:
                    res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun) and -res.fun > best_sum:
                        vals = constraints(res.x)
                        if np.min(vals) >= -1e-6:
                            best_x = res.x.copy()
                            best_sum = -res.fun
                except Exception:
                    pass
                    
                # Deflation strategy: shrink radii slightly to allow center rearrangement
                x_shrink = best_x.copy()
                x_shrink[2::3] *= 0.94
                for i in range(N):
                    r = x_shrink[3*i+2]
                    x_shrink[3*i] = np.clip(x_shrink[3*i], r, 1.0-r)
                    x_shrink[3*i+1] = np.clip(x_shrink[3*i+1], r, 1.0-r)
                    
                try:
                    res = minimize(objective, x_shrink, method='SLSQP', bounds=bounds,
                                   constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                    if not np.isnan(res.fun) and -res.fun > best_sum:
                        vals = constraints(res.x)
                        if np.min(vals) >= -1e-6:
                            best_x = res.x.copy()
                            best_sum = -res.fun
                except Exception:
                    pass
                    
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Final safety shrink to guarantee strict compliance
    for _ in range(100):
        valid = True
        for i in range(N):
            if radii[i] < 0: valid=False; break
            if centers[i,0]-radii[i] < -1e-9 or centers[i,0]+radii[i] > 1+1e-9: valid=False; break
            if centers[i,1]-radii[i] < -1e-9 or centers[i,1]+radii[i] > 1+1e-9: valid=False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i]+radii[j]-1e-9:
                        valid=False; break
                if not valid: break
        if valid: break
        radii *= 0.995
        
    return centers, radii, float(np.sum(radii))
