# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=09c39892 sum of radii=1.473605 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*n:])


def constraint_func(v, n, pair_i, pair_j):
    """
    Vectorized inequality constraints using squared distances.
    Returns array of values that must be >= 0.
    """
    centers_x = v[:n]
    centers_y = v[n:2*n]
    radii = v[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = []
    cons.append(centers_x - radii)           # x - r >= 0
    cons.append(1.0 - centers_x - radii)    # 1 - x - r >= 0
    cons.append(centers_y - radii)           # y - r >= 0
    cons.append(1.0 - centers_y - radii)    # 1 - y - r >= 0
    
    # Pairwise non-overlap: dist_sq >= (r_i + r_j)^2
    cx_i = centers_x[pair_i]
    cx_j = centers_x[pair_j]
    cy_i = centers_y[pair_i]
    cy_j = centers_y[pair_j]
    r_i = radii[pair_i]
    r_j = radii[pair_j]
    
    dist_sq = (cx_i - cx_j)**2 + (cy_i - cy_j)**2
    r_sum_sq = (r_i + r_j)**2
    
    cons.append(dist_sq - r_sum_sq)
    
    return np.concatenate(cons)


def get_pair_indices(n):
    """Precompute indices for all unique circle pairs."""
    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)


def init_grid(n, seed=None):
    """Grid-based initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((n, 2))
    # 6x5 grid layout
    count = 0
    for r in range(6):
        for c in range(5):
            if count < n:
                x = 0.09 + c * 0.18
                y = 0.09 + r * 0.165
                centers[count] = [x, y]
                count += 1
    
    # Shuffle
    perm = np.random.permutation(n)
    centers = centers[perm]
    
    # Add jitter
    centers += np.random.uniform(-0.015, 0.015, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    return centers


def init_hex(n, seed=None):
    """Hexagonal lattice initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    r_init = 0.085
    centers = []
    y = r_init
    row = 0
    
    while len(centers) < n + 5:
        x_start = r_init + (row % 2) * r_init
        x = x_start
        while x <= 1.0 - r_init and len(centers) < n + 5:
            centers.append([x, y])
            x += 2.0 * r_init
        y += r_init * np.sqrt(3) * 0.95
        row += 1
    
    centers = np.array(centers[:n])
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    return centers


def init_random_dense(n, seed=None):
    """Random dense initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.random.uniform(0.1, 0.9, size=(n, 2))
    return centers


def run_optimization(n, centers_init, r_init, pair_i, pair_j, bounds, max_iter=3000, tol=1e-12):
    """Run single optimization with given initial configuration."""
    x0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], r_init])
    
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n, pair_i, pair_j)}
    
    try:
        res = minimize(objective_func, x0, method='SLSQP',
                       bounds=bounds, args=(n,),
                       constraints=cons,
                       options={'maxiter': max_iter, 'ftol': tol, 'disp': False})
        return res
    except Exception:
        return None


def validate_and_fix(n, centers, radii, pair_i, pair_j):
    """
    Post-process to ensure strict validity.
    Iteratively shrink radii until all constraints are satisfied.
    """
    radii = radii.copy()
    
    # Ensure boundary constraints
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max(0.0, max_r))
    
    # Iteratively fix overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            # Boundary
            max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                         centers[i, 1], 1.0 - centers[i, 1])
            
            # Neighbor constraints
            for j_idx, j in enumerate(range(n)):
                if i == j:
                    continue
                dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                               (centers[i, 1] - centers[j, 1])**2)
                max_r = min(max_r, max(0.0, dist - radii[j]))
            
            if radii[i] > max_r + 1e-12:
                radii[i] = max(0.0, max_r)
                changed = True
        
        if not changed:
            break
    
    return radii


def run_packing():
    n = 26
    pair_i, pair_j = get_pair_indices(n)
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
    for _ in range(n):
        bounds.append((0.0, 1.0))  # y
    for _ in range(n):
        bounds.append((0.0, 0.5))   # r
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Strategy 1: Grid-based initializations with multiple seeds
    for seed in range(6):
        centers_init = init_grid(n, seed=seed)
        r_init = np.full(n, 0.04)
        
        res = run_optimization(n, centers_init, r_init, pair_i, pair_j, bounds,
                               max_iter=4000, tol=1e-12)
        
        if res is not None and res.success:
            centers_cand = res.x[:2*n].reshape(n, 2)
            radii_cand = res.x[2*n:]
            radii_cand = validate_and_fix(n, centers_cand, radii_cand, pair_i, pair_j)
            
            current_sum = np.sum(radii_cand)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers_cand.copy()
                best_radii = radii_cand.copy()
    
    # Strategy 2: Hexagonal-based initializations
    for seed in range(5):
        centers_init = init_hex(n, seed=seed)
        r_init = np.full(n, 0.045)
        
        res = run_optimization(n, centers_init, r_init, pair_i, pair_j, bounds,
                               max_iter=4000, tol=1e-12)
        
        if res is not None and res.success:
            centers_cand = res.x[:2*n].reshape(n, 2)
            radii_cand = res.x[2*n:]
            radii_cand = validate_and_fix(n, centers_cand, radii_cand, pair_i, pair_j)
            
            current_sum = np.sum(radii_cand)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers_cand.copy()
                best_radii = radii_cand.copy()
    
    # Strategy 3: Fine-tuning from best solution found so far
    if best_centers is not None:
        # Perturb best solution slightly and re-optimize
        for seed in range(4):
            np.random.seed(seed + 100)
            centers_pert = best_centers + np.random.uniform(-0.005, 0.005, size=best_centers.shape)
            centers_pert = np.clip(centers_pert, 0.05, 0.95)
            r_init = best_radii * 0.95
            
            res = run_optimization(n, centers_pert, r_init, pair_i, pair_j, bounds,
                                   max_iter=3000, tol=1e-13)
            
            if res is not None:
                centers_cand = res.x[:2*n].reshape(n, 2)
                radii_cand = res.x[2*n:]
                radii_cand = validate_and_fix(n, centers_cand, radii_cand, pair_i, pair_j)
                
                current_sum = np.sum(radii_cand)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers_cand.copy()
                    best_radii = radii_cand.copy()
    
    # Strategy 4: Try with slightly different radius initialization values
    if best_centers is not None:
        centers_base = best_centers.copy()
        for r_mult in [0.85, 0.9, 0.95]:
            r_init = best_radii * r_mult
            centers_pert = centers_base + np.random.uniform(-0.008, 0.008, size=centers_base.shape)
            centers_pert = np.clip(centers_pert, 0.05, 0.95)
            
            res = run_optimization(n, centers_pert, r_init, pair_i, pair_j, bounds,
                                   max_iter=3500, tol=1e-13)
            
            if res is not None:
                centers_cand = res.x[:2*n].reshape(n, 2)
                radii_cand = res.x[2*n:]
                radii_cand = validate_and_fix(n, centers_cand, radii_cand, pair_i, pair_j)
                
                current_sum = np.sum(radii_cand)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers_cand.copy()
                    best_radii = radii_cand.copy()
    
    # Final validation
    if best_centers is not None and best_radii is not None:
        final_radii = validate_and_fix(n, best_centers, best_radii, pair_i, pair_j)
        best_sum = np.sum(final_radii)
        
        # Safety margin to ensure strict validity
        final_radii *= 0.9999999
    
    if best_centers is None:
        # Fallback
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.06)
        count = 0
        for r in range(6):
            for c in range(5):
                if count < n:
                    best_centers[count] = [0.1 + c*0.17, 0.1 + r*0.16]
                    count += 1
        best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
