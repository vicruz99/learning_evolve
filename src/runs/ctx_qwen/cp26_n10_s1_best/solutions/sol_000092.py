# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000072 (state e356f834) state=b7bcce0d sum of radii=2.623489 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def get_constraints(x):
    """Compute all boundary and non-overlap constraints as a vector >= 0."""
    cx, cy, cr = x[0::3], x[1::3], x[2::3]
    
    # Boundary constraints: 4 * N
    c = np.concatenate([
        cx - cr,
        1.0 - cx - cr,
        cy - cr,
        1.0 - cy - cr
    ])
    
    # Overlap constraints: N*(N-1)/2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dr = cr[:, None] + cr[None, :]
    
    i_idx, j_idx = np.tril_indices(N_CIRCLES, -1)
    dist_sq = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2
    r_sum_sq = dr[i_idx, j_idx]**2
    
    c = np.concatenate([c, dist_sq - r_sum_sq])
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N_CIRCLES):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def solve_lp_radii(centers):
    """Optimally compute radii for fixed centers using Linear Programming."""
    n = N_CIRCLES
    c_obj = -np.ones(n)
    num_ineq = n + n * (n - 1) // 2
    A_ub = np.zeros((num_ineq, n))
    b_ub = np.zeros(num_ineq)
    
    idx = 0
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        A_ub[idx, i] = 1.0
        b_ub[idx] = max(0.0, lim)
        idx += 1
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
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
            return res.x
    except Exception:
        pass
    return None

def make_hex_init(r0, angle, seed):
    """Generate a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    n = N_CIRCLES
    centers = []
    y = r0
    row = 0
    while len(centers) < n + 10:
        x_start = r0 + (r0 if row % 2 == 1 else 0.0)
        x = x_start
        while x <= 1.0 - r0 and len(centers) < n + 10:
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    centers = np.array(centers[:n + 10])
    
    # Rotate around center
    if angle != 0.0:
        c = np.array([0.5, 0.5])
        ca, sa = np.cos(angle), np.sin(angle)
        centers = ((centers - c) @ np.array([[ca, -sa], [sa, ca]]) + c)
        
    # Filter points strictly inside the square
    mask = (centers[:, 0] >= 0.01) & (centers[:, 0] <= 0.99) & \
           (centers[:, 1] >= 0.01) & (centers[:, 1] <= 0.99)
    centers = centers[mask]
    
    # Pad if rotation/filtering removed points
    while len(centers) < n:
        centers = np.vstack([centers, [np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)]])
    centers = centers[:n]
    
    # Perturb and ensure initial feasibility relative to r0
    centers += np.random.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, r0 + 0.001, 1.0 - r0 - 0.001)
    
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = r0
    return x0

def run_packing():
    n = N_CIRCLES
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Phase 1: Diverse multi-start optimization
    inits = []
    # Varied hexagonal starts to cover different density basins and orientations
    for s in range(30):
        r0 = 0.085 + s * 0.0015
        inits.append(make_hex_init(r0, angle=s * 0.04, seed=s * 100))
        
    for x0 in inits:
        # Use LP to find optimal radii for these centers initially
        c_init = np.column_stack((x0[0::3], x0[1::3]))
        r_lp = solve_lp_radii(c_init)
        if r_lp is not None:
            x0[2::3] = r_lp
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            c_vals = get_constraints(res.x)
            # Accept if feasible within tolerance and improves best sum
            if np.min(c_vals) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative Deflation-Reinforcement Refinement
    if best_x is not None:
        for rnd in range(20):
            x0 = best_x.copy()
            # Shrink radii slightly to allow centers to move into denser configurations
            x0[2::3] *= 0.985
            
            noise_scale = 0.0025 * (0.90 ** rnd)
            x0[0::3] += np.random.normal(0, noise_scale, n)
            x0[1::3] += np.random.normal(0, noise_scale, n)
            
            # Project perturbed variables back to strict bounds
            for k in range(n):
                r = max(0.005, x0[3*k + 2])
                x0[3*k] = np.clip(x0[3*k], r, 1.0 - r)
                x0[3*k + 1] = np.clip(x0[3*k + 1], r, 1.0 - r)
                x0[3*k + 2] = r
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False})
                curr_sum = -res.fun
                c_vals = get_constraints(res.x)
                if np.min(c_vals) >= -1e-6 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
                    
                    # LP refinement on current centers to squeeze out extra radius
                    c_opt = np.column_stack((best_x[0::3], best_x[1::3]))
                    r_new = solve_lp_radii(c_opt)
                    if r_new is not None:
                        best_x[2::3] = r_new
                        best_sum = np.sum(r_new)
            except Exception:
                pass
                
    # Extract centers and radii
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    # Phase 3: Strict validity check and numerical repair
    for _ in range(100):
        valid = True
        for i in range(n):
            if (radii[i] < 0 or 
                centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1.0 - radii[i] + 1e-9 or 
                centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1.0 - radii[i] + 1e-9):
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            break
            
        # Minimal shrinkage to guarantee strict compliance
        radii *= 0.999
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
    return centers, radii, float(np.sum(radii))
