# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000002 (state 2c120403) state=69bc282d sum of radii=2.626129 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def compute_overlap(p):
    """Computes squared distance minus squared sum of radii for all pairs (i < j)."""
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    sum_r_sq = dr**2
    
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    return (dist_sq - sum_r_sq)[mask]

def compute_boundary(p):
    """Computes boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0, r >= 0."""
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r, r])

def objective(p):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def make_bounds():
    """Creates variable bounds for x, y in [0,1] and r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def get_hex_init(r=0.095, noise=0.0):
    """Generates initial parameters using a hexagonal lattice pattern."""
    params = np.zeros(N * 3)
    idx = 0
    y = r
    row = 0
    while idx < N:
        start_x = r if row % 2 == 0 else 2 * r
        x = start_x
        while x + r <= 1.0 + 1e-9 and idx < N:
            params[3*idx] = x
            params[3*idx+1] = y
            params[3*idx+2] = r
            idx += 1
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
        
    if noise > 0:
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        params[0::3] += rng.normal(0, noise, N)
        params[1::3] += rng.normal(0, noise, N)
        params[0::3] = np.clip(params[0::3], 1e-4, 1 - 1e-4)
        params[1::3] = np.clip(params[1::3], 1e-4, 1 - 1e-4)
    return params

def get_grid_init(r=0.085, noise=0.0):
    """Generates initial parameters using a square grid pattern."""
    params = np.zeros(N * 3)
    idx = 0
    cols = 6
    rows = 5
    for row in range(rows):
        for col in range(cols):
            if idx >= N:
                break
            x = r + col * 2 * r
            y = r + row * 2 * r
            params[3*idx] = x
            params[3*idx+1] = y
            params[3*idx+2] = r
            idx += 1
            
    if noise > 0:
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        params[0::3] += rng.normal(0, noise, N)
        params[1::3] += rng.normal(0, noise, N)
        params[0::3] = np.clip(params[0::3], 1e-4, 1 - 1e-4)
        params[1::3] = np.clip(params[1::3], 1e-4, 1 - 1e-4)
    return params

def get_random_init(seed, r=0.06):
    """Generates random initial parameters."""
    rng = np.random.default_rng(seed)
    params = np.zeros(N * 3)
    params[0::3] = rng.uniform(0.15, 0.85, N)
    params[1::3] = rng.uniform(0.15, 0.85, N)
    params[2::3] = r
    return params

def run_optimization(p0, bounds, constraints):
    """Runs SLSQP optimization and returns optimized parameters and success flag."""
    try:
        res = opt.minimize(
            objective,
            p0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False}
        )
        return res.x, res.success
    except Exception:
        return p0, False

def repair_solution(centers, radii):
    """Iteratively shrinks radii to resolve overlaps and clamp to boundaries."""
    n = centers.shape[0]
    radii = radii.copy()
    
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    overlap = sum_r - dist
                    total_r = radii[i] + radii[j]
                    if total_r > 1e-12:
                        shrink_i = overlap * (radii[i] / total_r)
                        shrink_j = overlap * (radii[j] / total_r)
                    else:
                        shrink_i = overlap / 2.0
                        shrink_j = overlap / 2.0
                        
                    radii[i] -= shrink_i
                    radii[j] -= shrink_j
                    changed = True
        if not changed:
            break
            
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            radii[i] = max_r
            
    radii = np.maximum(radii, 0.0)
    return radii

def run_packing() -> tuple:
    """Main function to pack 26 circles in a unit square."""
    np.random.seed(42)
    bounds = make_bounds()
    cons_overlap = opt.NonlinearConstraint(compute_overlap, 0, np.inf)
    cons_boundary = opt.NonlinearConstraint(compute_boundary, 0, np.inf)
    constraints = [cons_overlap, cons_boundary]
    
    best_p = None
    best_sum = -1.0
    
    # Diverse initial configurations
    configs = [
        ("hex", 0.095, 0.0),
        ("hex", 0.095, 0.005),
        ("hex", 0.090, 0.01),
        ("grid", 0.085, 0.005),
        ("grid", 0.085, 0.01),
        ("rand", 10, 0.06),
        ("rand", 20, 0.06),
        ("rand", 30, 0.05),
        ("rand", 40, 0.05),
        ("rand", 50, 0.045),
    ]
    
    for cfg in configs:
        typ = cfg[0]
        if typ == "hex":
            p0 = get_hex_init(r=cfg[1], noise=cfg[2])
        elif typ == "grid":
            p0 = get_grid_init(r=cfg[1], noise=cfg[2])
        else:
            p0 = get_random_init(seed=cfg[1], r=cfg[2])
            
        # Tiny deterministic perturbation to break symmetry
        p0[0::3] += np.random.normal(0, 1e-4, N)
        p0[1::3] += np.random.normal(0, 1e-4, N)
        
        p_opt, success = run_optimization(p0, bounds, constraints)
        
        if success:
            r_opt = p_opt[2::3]
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                centers_temp = np.column_stack((p_opt[0::3], p_opt[1::3]))
                r_temp = repair_solution(centers_temp, r_opt)
                repaired_sum = np.sum(r_temp)
                if repaired_sum > best_sum:
                    best_sum = repaired_sum
                    best_p = p_opt.copy()
                    best_p[0::3] = centers_temp[:, 0]
                    best_p[1::3] = centers_temp[:, 1]
                    best_p[2::3] = r_temp

    if best_p is None:
        best_p = get_hex_init(0.09)
        
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    
    centers = np.clip(centers, 0.0, 1.0)
    radii = repair_solution(centers, radii)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
