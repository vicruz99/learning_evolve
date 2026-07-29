# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 33c0c451) state=08bdb78f sum of radii=1.576057 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N_CIRCLES = 26
I_U, J_U = np.triu_indices(N_CIRCLES, k=1)

def objective_joint(x):
    """Minimize negative sum of radii"""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints g(x) >= 0 for SLSQP"""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Squared distance constraints: dist^2 - (r_i + r_j)^2 >= 0
    dx = cx[I_U] - cx[J_U]
    dy = cy[I_U] - cy[J_U]
    dist_sq = dx*dx + dy*dy
    r_sum = r[I_U] + r[J_U]
    c_pair = dist_sq - r_sum**2
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_pair, c_bound])

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii"""
    n = centers.shape[0]
    c_obj = -np.ones(n)  # Maximize sum(r) -> Minimize -sum(r)
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    A_ub = np.zeros((n * (n - 1) // 2, n))
    b_ub = np.zeros(n * (n - 1) // 2)
    
    # Precompute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    # Boundary constraints handled via variable bounds
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(max_r, 0.0)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    # Fallback if LP fails
    return np.zeros(n), 0.0

def create_hex_initial(seed, spacing_range=(0.14, 0.20)):
    """Generate a hexagonal lattice initialization"""
    rng = np.random.RandomState(seed)
    centers = []
    
    # Randomize hex parameters slightly per seed
    r_init = 0.03 + rng.uniform(0.0, 0.02)
    y_step = r_init * math.sqrt(3) + rng.uniform(-0.005, 0.005)
    x_step = 2 * r_init + rng.uniform(-0.005, 0.005)
    y_start = r_init + rng.uniform(0.0, 0.02)
    x_start = r_init + rng.uniform(0.0, 0.02)
    
    y = y_start
    row = 0
    while len(centers) < N_CIRCLES:
        shift = 0.0 if row % 2 == 0 else x_step / 2.0
        x = x_start + shift
        while x <= 1.0 - r_init and len(centers) < N_CIRCLES:
            centers.append([x, y])
            x += x_step
        y += y_step
        row += 1
        
    centers = np.array(centers[:N_CIRCLES])
    # Add small noise
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    # Strategy 1: Hexagonal lattice variations
    for seed in range(20):
        init_centers = create_hex_initial(seed)
        r_init = np.full(N_CIRCLES, 0.02)
        x0 = np.concatenate([init_centers.ravel(), r_init])
        
        try:
            res = minimize(
                objective_joint, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                    best_radii = res.x[2*N_CIRCLES:]
        except Exception:
            continue

    # Strategy 2: Random restarts with careful bounds
    for seed in range(15):
        rng = np.random.RandomState(seed)
        cx = rng.uniform(0.15, 0.85, N_CIRCLES)
        cy = rng.uniform(0.15, 0.85, N_CIRCLES)
        r_init = np.full(N_CIRCLES, 0.02)
        x0 = np.concatenate([cx, cy, r_init])
        
        try:
            res = minimize(
                objective_joint, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                    best_radii = res.x[2*N_CIRCLES:]
        except Exception:
            continue

    # LP Refinement on best centers found
    if best_centers is not None:
        refined_radii, refined_sum = solve_radii_lp(best_centers)
        if refined_sum > 0.0:
            best_radii = refined_radii
            best_sum = refined_sum
            
        # Ensure numerical safety: slight shrink if any constraint is barely violated
        centers = best_centers.copy()
        radii = best_radii.copy()
        
        # Boundary safety
        for i in range(N_CIRCLES):
            x, y = centers[i]
            r = radii[i]
            margin = 1e-9
            max_r_bound = min(x, 1-x, y, 1-y) - margin
            if radii[i] > max_r_bound:
                radii[i] = max_r_bound
                
        # Overlap safety: iterative reduction
        for _ in range(50):
            changed = False
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-10:
                        overlap = (radii[i] + radii[j] - dist) / 2.0 + 1e-9
                        radii[i] -= overlap
                        radii[j] -= overlap
                        changed = True
            if not changed:
                break
                
        radii = np.maximum(radii, 0.0)
        best_sum = np.sum(radii)
        
    else:
        # Absolute fallback
        centers = np.tile([0.5, 0.5], (N_CIRCLES, 1))
        radii = np.full(N_CIRCLES, 0.02)
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii

    return best_centers, best_radii, float(best_sum)
