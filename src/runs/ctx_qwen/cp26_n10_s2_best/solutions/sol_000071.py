# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=d20b2c8c sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Objective function to minimize: negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def constraint_func(v, idx_i, idx_j):
    """
    Vectorized constraint function.
    Returns an array where all elements must be >= 0.
    Uses squared distances for numerical stability.
    """
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    # Boundary constraints: center +/- radius within [0, 1]
    c_boundary = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    c1 = centers[idx_i]
    c2 = centers[idx_j]
    diff = c1 - c2
    dist_sq = np.sum(diff**2, axis=1)
    r_sum = radii[idx_i] + radii[idx_j]
    c_overlap = dist_sq - r_sum**2
    
    return np.concatenate([c_boundary, c_overlap])

def generate_init(seed, init_type='hex'):
    """Generates an initial valid configuration."""
    np.random.seed(seed)
    centers = np.zeros((N_CIRCLES, 2))
    
    if init_type == 'hex':
        r_est = 0.09
        pts = []
        y = r_est
        row = 0
        while len(pts) < N_CIRCLES + 10:
            x_start = r_est + (row % 2) * r_est
            x = x_start
            while x <= 1.0 - r_est and len(pts) < N_CIRCLES + 10:
                pts.append([x, y])
                x += 2.0 * r_est
            y += np.sqrt(3) * r_est
            row += 1
        centers = np.array(pts[:N_CIRCLES])
    elif init_type == 'grid':
        xs = np.linspace(0.12, 0.88, 6)
        ys = np.linspace(0.12, 0.88, 5)
        pts = []
        for y in ys:
            for x in xs:
                pts.append([x, y])
        centers = np.array(pts[:N_CIRCLES])
    else:  # random
        centers = np.random.uniform(0.1, 0.9, (N_CIRCLES, 2))
        
    # Perturb centers slightly
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    # Compute safe initial radii based on local geometry
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Radii limited by walls and neighbors
    wall_limits = np.minimum(centers[:,0], 1 - centers[:,0])
    wall_limits = np.minimum(wall_limits, np.minimum(centers[:,1], 1 - centers[:,1]))
    radii = np.minimum(min_dists / 2.0 - 0.002, wall_limits)
    radii = np.clip(radii, 0.01, 0.25)
    
    # Add random variation to radii to break symmetry
    radii *= (0.9 + 0.2 * np.random.rand(N_CIRCLES))
    
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    # Precompute pair indices for efficiency
    idx_i, idx_j = np.triu_indices(N_CIRCLES, k=1)
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    
    # Constraint definition
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (idx_i, idx_j)}
    
    best_sum = -1.0
    best_v = None
    
    # Generate diverse starting configurations
    inits = []
    for s in range(12):
        inits.append(('hex', s))
        inits.append(('grid', s))
        inits.append(('rand', s))
        
    for itype, seed in inits:
        v0 = generate_init(seed, itype)
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            current_sum = -res.fun
            # Check feasibility with tolerance
            if np.all(constraint_func(res.x, idx_i, idx_j) >= -1e-7) and current_sum > best_sum:
                best_sum = current_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if all restarts fail
    if best_v is None:
        v0 = generate_init(0, 'hex')
        res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 4000})
        best_v = res.x
        best_sum = -res.fun
        
    centers = best_v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_v[2*N_CIRCLES:]
    
    # Strict post-processing to guarantee validity
    # 1. Enforce boundary constraints
    for i in range(N_CIRCLES):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    # 2. Iteratively fix overlaps
    for _ in range(20):
        stable = True
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    shrink = (sum_r - dist) / 2.0 + 1e-8
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    stable = False
        if stable:
            break
            
    return centers, radii, float(np.sum(radii))
