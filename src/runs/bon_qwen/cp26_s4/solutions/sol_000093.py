# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=fd97a2ed sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_objective(params):
    # Objective: maximize sum of radii => minimize negative sum
    return -np.sum(params[2::3])

def compute_constraints(params):
    # Returns inequality constraints g(x) >= 0
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    n = len(rs)
    total_c = 4 * n + n * (n - 1) // 2
    c = np.empty(total_c)
    idx = 0
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    for i in range(n):
        c[idx]   = xs[i] - rs[i]
        c[idx+1] = 1.0 - xs[i] - rs[i]
        c[idx+2] = ys[i] - rs[i]
        c[idx+3] = 1.0 - ys[i] - rs[i]
        idx += 4
        
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy
    r_sum = rs[:, None] + rs[None, :]
    r_sum2 = r_sum * r_sum
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c[idx:] = (d2[mask] - r_sum2[mask]).flatten()
    return c

def refine_radii_lp(centers):
    # Given fixed centers, find optimal radii via Linear Programming
    n = N_CIRCLES
    c_obj = -np.ones(n)  # Minimize -sum(r) <=> Maximize sum(r)
    A_list = []
    b_list = []
    
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        # r_i <= x_i
        A_list.append(row); b_list.append(centers[i, 0])
        # r_i <= 1 - x_i
        A_list.append(row); b_list.append(1.0 - centers[i, 0])
        # r_i <= y_i
        A_list.append(row); b_list.append(centers[i, 1])
        # r_i <= 1 - y_i
        A_list.append(row); b_list.append(1.0 - centers[i, 1])
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            # r_i + r_j <= d
            A_list.append(row); b_list.append(d)
            
    A_ub = np.array(A_list)
    b_ub = np.array(b_list)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def run_packing():
    best_sum = -1.0
    best_centers = np.zeros((N_CIRCLES, 2))
    best_radii = np.zeros(N_CIRCLES)
    
    con = {'type': 'ineq', 'fun': compute_constraints}
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N_CIRCLES
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Perturbed hexagonal grids with varying scales/offsets
    for seed in range(8):
        np.random.seed(seed)
        grid = []
        y = 0.06
        row = 0
        scale = 0.16 + 0.02 * (seed % 3)
        while y < 0.94:
            x = 0.06 + (row % 2) * 0.5 * scale
            while x < 0.94:
                grid.append([x + np.random.uniform(-0.02, 0.02), 
                             y + np.random.uniform(-0.02, 0.02)])
                x += scale
            y += scale * np.sqrt(3) / 2
            row += 1
        pts = np.array(grid[:N_CIRCLES])
        pts = np.clip(pts, 0.05, 0.95)
        p = np.zeros(3 * N_CIRCLES)
        p[0::3] = pts[:, 0]
        p[1::3] = pts[:, 1]
        p[2::3] = np.random.uniform(0.03, 0.06, N_CIRCLES)
        inits.append(p)
        
    # 2. Random dense packings
    for seed in range(12):
        np.random.seed(200 + seed)
        c = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
        r = np.random.uniform(0.02, 0.07, N_CIRCLES)
        p = np.zeros(3 * N_CIRCLES)
        p[0::3] = c[:, 0]
        p[1::3] = c[:, 1]
        p[2::3] = r
        inits.append(p)
        
    # Optimization loop
    for i, p0 in enumerate(inits):
        try:
            res = minimize(compute_objective, p0, method='SLSQP', bounds=bounds, 
                           constraints=con, options={'maxiter': 1000, 'ftol': 1e-11})
            
            c_opt = res.x[0::3].reshape(N_CIRCLES, 2)
            r_opt = res.x[2::3]
            
            # Phase 2: Exact radius optimization via LP
            r_lp = refine_radii_lp(c_opt)
            if r_lp is not None:
                r_opt = r_lp
                
            cur_sum = np.sum(r_opt)
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
            
    # Final numerical safety clamping
    best_radii = np.maximum(best_radii, 0.0)
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    return best_centers, best_radii, best_sum
