# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=5dd91b29 sum of radii=2.619889 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute upper triangle indices for pairwise constraints (i < j)
pair_i, pair_j = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Vectorized inequality constraints.
    Returns array where all elements must be >= 0.
    Uses squared distances for smoother gradients.
    """
    centers = v[:2*N].reshape(N, 2)
    radii = v[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    c_i = centers[pair_i]
    c_j = centers[pair_j]
    r_i = radii[pair_i]
    r_j = radii[pair_j]
    
    dist_sq = np.sum((c_i - c_j)**2, axis=1)
    r_sum_sq = (r_i + r_j)**2
    
    return np.concatenate([cons, dist_sq - r_sum_sq])

def generate_initial_guess(seed, strategy):
    """Generates a valid initial configuration based on strategy."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if strategy == 0:
        # Hexagonal lattice packing
        r0 = 0.095
        y = r0
        row = 0
        idx = 0
        while idx < N:
            start_x = r0 if row % 2 == 0 else 2 * r0
            x = start_x
            while x <= 1 - r0 and idx < N:
                centers[idx] = [
                    x + np.random.uniform(-0.02, 0.02),
                    y + np.random.uniform(-0.02, 0.02)
                ]
                x += 2 * r0
                idx += 1
            y += r0 * np.sqrt(3)
            row += 1
    else:
        # Perturbed grid packing
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < N:
                    centers[idx] = [
                        0.1 + c * 0.2 + np.random.uniform(-0.03, 0.03),
                        0.1 + r * 0.2 + np.random.uniform(-0.03, 0.03)
                    ]
                    idx += 1
        while idx < N:
            centers[idx] = np.random.uniform(0.1, 0.9, 2)
            idx += 1
            
    # Initialize with small valid radii
    return np.concatenate([centers.flatten(), np.full(N, 0.04)])

def run_packing():
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_x = None
    
    # Multi-start optimization with diverse strategies and seeds
    for seed in range(15):
        strategy = 0 if seed % 3 == 0 else 1
        try:
            x0 = generate_initial_guess(seed, strategy)
            
            # SLSQP handles bounds and nonlinear inequality constraints well
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, 
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            
            # Accept if strictly feasible and better
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) > -1e-6:
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization completely fails (unlikely)
    if best_x is None:
        best_x = generate_initial_guess(0, 0)
        
    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Post-processing: Enforce strict boundary constraints
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], 
                    centers[i, 1], 1 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        
    # Post-processing: Iteratively resolve overlaps due to numerical drift
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                sum_r = radii[i] + radii[j]
                if d < sum_r - 1e-9:
                    shrink = (sum_r - d) / 2.0 + 1e-7
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
