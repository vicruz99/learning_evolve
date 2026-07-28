# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=2513a577 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """Compute inequality constraints: returns array where each element must be >= 0."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    b = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Using squared distances provides smooth, well-conditioned gradients for SLSQP
    i, j = np.triu_indices(N, k=1)
    dx = cx[i] - cx[j]
    dy = cy[i] - cy[j]
    p = dx**2 + dy**2 - (r[i] + r[j])**2
    
    return np.concatenate([b, p])

def lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 0.0)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    num_pairs = N * (N - 1) // 2
    num_constraints = num_pairs + 4 * N
    A_ub = np.zeros((num_constraints, N))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    for i in range(N):
        for val in lims[i]:
            A_ub[idx, i] = 1.0
            b_ub[idx] = val
            idx += 1
            
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(N, 0.0), 0.0

def run_packing():
    np.random.seed(42)
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.2)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    configs = []
    
    # 1. Hexagonal lattice initialization
    r_init = 0.09
    pts = []
    y = r_init
    row = 0
    while len(pts) < N + 10:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0:
            pts.append([x, y])
            x += 2.0 * r_init
        y += r_init * np.sqrt(3)
        row += 1
    base_hex = np.array(pts[:N])
    configs.append(base_hex)
    
    # 2. Systematic perturbations of hex grid
    for p in [0.01, 0.02, 0.04, 0.06, 0.08, 0.12]:
        configs.append(np.clip(base_hex + np.random.uniform(-p, p, base_hex.shape), 0.05, 0.95))
        
    # 3. Random dense configurations to escape structural biases
    for _ in range(20):
        configs.append(np.random.uniform(0.1, 0.9, (N, 2)))
        
    # Multi-start optimization
    for cfg in configs:
        x0 = np.zeros(3 * N)
        x0[0::3] = cfg[:, 0]
        x0[1::3] = cfg[:, 1]
        x0[2::3] = 0.05  # Start with feasible small radii
        
        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds_vars,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            # Accept if successful or found a significantly large sum
            if res.success or res.fun < -2.5:
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt, s_opt = lp_radii(c_opt)
                
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Fallback if optimization yields poor results
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(N, 0.08)
        best_sum = np.sum(best_radii)
        
    # Final safety shrink to guarantee strict numerical validity against 1e-12 tolerance
    best_radii *= 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
