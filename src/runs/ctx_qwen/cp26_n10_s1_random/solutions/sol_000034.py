# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state e14e8c08) state=e427cf82 sum of radii=2.609039 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars_flat):
    """Minimize negative sum of radii."""
    return -np.sum(vars_flat[2::3])

def constraints(vars_flat):
    """Compute inequality constraints: returns array where each element must be >= 0."""
    x = vars_flat[0::3]
    y = vars_flat[1::3]
    r = vars_flat[2::3]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    b = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist(i,j) - (r_i + r_j) >= 0
    # Using Euclidean distance for better gradient conditioning near contact
    i, j = np.triu_indices(N, k=1)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    p = np.sqrt(dx**2 + dy**2) - (r[i] + r[j])
    
    return np.concatenate([b, p])

def run_packing():
    # Variable bounds: x,y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    pts1 = []
    y = 0.06
    row = 0
    while len(pts1) < N + 5:
        x = 0.06
        shift = 0.06 if row % 2 == 1 else 0.0
        while x < 0.94:
            pts1.append([x + shift, y])
            x += 0.12
        y += 0.104
        row += 1
    configs.append(np.array(pts1[:N]))
    
    # 2. Perturbed Hex
    configs.append(np.clip(configs[0] + np.random.uniform(-0.02, 0.02, (N, 2)), 0.05, 0.95))
    
    # 3. Dense Random
    configs.append(np.random.uniform(0.1, 0.9, size=(N, 2)))
    
    # 4. Structured Grid 5x5 + 1
    grid = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for j in range(5) for i in range(5)])
    grid = np.vstack([grid, [[0.5, 0.9]]])
    configs.append(grid)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    r_init = 0.06
    
    # Run SLSQP on each configuration
    for cfg in configs:
        x0 = np.zeros(3 * N)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
            
            # Accept if successful or significantly improved
            if res.success or res.fun < -2.5:
                centers = np.column_stack((res.x[0::3], res.x[1::3]))
                radii = res.x[2::3]
                
                # Verify feasibility strictly
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6:
                    s = np.sum(radii)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers.copy()
                        best_radii = radii.copy()
        except Exception:
            continue
            
    # Fallback to grid if optimization yields poor results
    if best_centers is None or best_sum < 2.0:
        best_centers = configs[3]
        best_radii = np.full(N, 0.09)
        best_sum = np.sum(best_radii)
        
    # Phase 2: Exact Linear Programming for radii given fixed optimal centers
    # Maximize sum(r_i) s.t. r_i + r_j <= dist(i,j) and boundary constraints
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    A_ub = np.zeros((num_pairs + 4 * N, N))
    b_ub = np.zeros(num_pairs + 4 * N)
    
    idx = 0
    for i, j in pairs:
        d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(N):
        x, y = best_centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            # Apply tiny shrink to guarantee strict validity against 1e-12 tolerance
            best_radii = lp_res.x * 0.9999999
            best_sum = np.sum(best_radii)
    except Exception:
        pass
        
    return best_centers, best_radii, float(best_sum)
