# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=f6ce444f sum of radii=2.628637 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to avoid heavy matrix operations
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dist - r_sum
    
    return np.concatenate([c_bound, c_pair])

def get_initial_config(seed, layout='hex'):
    """Generate a valid initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if layout == 'hex':
        r_est = 0.095
        pts = []
        y = r_est
        row = 0
        while len(pts) < N + 10:
            x_start = r_est + (row % 2) * r_est
            x = x_start
            while x <= 1 - r_est:
                pts.append([x, y])
                x += 2 * r_est
            y += np.sqrt(3) * r_est
            row += 1
        pts = np.array(pts)
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx]
    elif layout == 'grid':
        pts = np.array([[i*0.18 + 0.1, j*0.18 + 0.1] for i in range(6) for j in range(5)])
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx]
    else: # random
        centers = np.random.uniform(0.05, 0.95, size=(N, 2))
        
    # Add jitter and clip
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Varied initial radii to break symmetry, kept small for feasibility
    r_init = np.full(N, 0.04)
    r_init += np.random.uniform(-0.005, 0.005, N)
    r_init = np.clip(r_init, 0.02, 0.07)
    
    return np.concatenate([centers[:, 0], centers[:, 1], r_init])

def refine_search(v_best, seed, iterations=2500):
    """Perturb the best solution and run local optimization."""
    np.random.seed(seed)
    v_perturbed = v_best.copy()
    v_perturbed += np.random.uniform(-0.005, 0.005, size=v_best.shape)
    
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    try:
        res = minimize(objective, v_perturbed, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': iterations, 'ftol': 1e-12, 'disp': False})
        return res.x, -res.fun
    except Exception:
        return v_best, -np.sum(v_best[2*N:])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Stage 1: Multi-start exploration
    layouts = ['hex', 'grid', 'random']
    for layout in layouts:
        for seed in range(12):
            x0 = get_initial_config(seed, layout)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-11, 'disp': False})
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_v = res.x.copy()
            except Exception:
                pass
                
    if best_v is None:
        best_v = get_initial_config(0, 'hex')
        
    # Stage 2: Local refinement on the best configuration
    for seed in range(6):
        v_ref, s_ref = refine_search(best_v, seed, iterations=2500)
        if s_ref > best_sum:
            best_sum = s_ref
            best_v = v_ref
            
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # Strict post-processing to guarantee validation passes
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(10):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    # Shrink both equally to resolve overlap, add tiny buffer
                    shrink = excess / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    return centers, radii, float(np.sum(radii))
