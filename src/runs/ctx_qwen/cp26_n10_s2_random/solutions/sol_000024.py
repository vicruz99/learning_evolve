# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state 1cb5ec92) state=1a6d8cb5 sum of radii=2.591849 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def compute_constraints(v):
    """Compute all inequality constraint values (must be >= 0)."""
    n = len(v) // 3
    C = v[:2 * n].reshape(n, 2)
    R = v[2 * n:]
    
    constraints = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    # x >= r  => x - r >= 0
    constraints.append(C[:, 0] - R)
    # 1 - x >= r => 1 - x - r >= 0
    constraints.append(1.0 - C[:, 0] - R)
    # y >= r => y - r >= 0
    constraints.append(C[:, 1] - R)
    # 1 - y >= r => 1 - y - r >= 0
    constraints.append(1.0 - C[:, 1] - R)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    rsq = (R[:, np.newaxis] + R[np.newaxis, :])**2
    
    mask = np.triu_indices(n, k=1)
    constraints.append(d2[mask] - rsq[mask])
    
    # Non-negative radii
    constraints.append(R)
    
    return np.concatenate(constraints)

def objective_func(v):
    """Objective function: minimize negative sum of radii."""
    n = len(v) // 3
    return -np.sum(v[2 * n:])

def get_max_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= dist_ij
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        bounds_vals = [centers[i, 0], 1.0 - centers[i, 0], 
                       centers[i, 1], 1.0 - centers[i, 1]]
        for b in bounds_vals:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Solve LP
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                  bounds=(0.0, None), method='highs')
    
    if res.success:
        return res.x
    return np.full(n, 0.01)

def generate_initial_configs(n, num_configs):
    """Generate diverse initial center configurations."""
    configs = []
    
    # 1. Hexagonal lattice
    pts = []
    y = 0.15
    row = 0
    while len(pts) < n:
        x = 0.15 if row % 2 == 0 else 0.25
        while x <= 0.85 and len(pts) < n:
            pts.append([x, y])
            x += 0.25
        y += 0.2165  # approx sqrt(3)/2 * 0.25
        row += 1
    configs.append(np.array(pts[:n]))
    
    # 2. Random dense
    configs.append(np.random.rand(n, 2) * 0.8 + 0.1)
    
    # 3. Structured grid
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.12 + i * 0.2, 0.12 + j * 0.2])
    while len(pts) < n:
        pts.append([0.5, 0.5])
    configs.append(np.array(pts[:n]))
    
    # Fill remaining with perturbed hexagonal
    while len(configs) < num_configs:
        base = configs[0].copy()
        base += np.random.normal(0, 0.02, base.shape)
        base = np.clip(base, 0.05, 0.95)
        configs.append(base)
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in unit square to maximize sum of radii."""
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    
    np.random.seed(42)
    initial_configs = generate_initial_configs(n, 25)
    
    for centers_init in initial_configs:
        radii_init = np.full(n, 0.04)
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        try:
            # Non-linear optimization of centers and radii
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-12})
            
            # Extract optimized centers
            centers_opt = res.x[:2 * n].reshape(n, 2)
            
            # Exact LP refinement for radii given these centers
            radii_lp = get_max_radii_lp(centers_opt)
            current_sum = np.sum(radii_lp)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers_opt
                best_radii = radii_lp
        except Exception:
            continue
            
    # Fallback if all optimizations fail
    if best_centers is None:
        best_centers = initial_configs[0]
        best_radii = np.full(n, 0.04)
        best_sum = np.sum(best_radii)
        
    # Ensure numerical safety
    best_radii = np.maximum(best_radii, 1e-9)
    
    return best_centers, best_radii, float(best_sum)
