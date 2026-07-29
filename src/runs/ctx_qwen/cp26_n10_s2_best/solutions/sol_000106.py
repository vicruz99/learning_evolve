# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000055 (state f6ce444f) state=07810a19 sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to avoid heavy loops
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx*dx + dy*dy)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dist - r_sum
    
    return np.concatenate([c_bound, c_pair])

def compute_feasible_radii(centers, shrink_factor=0.95):
    """Compute strictly feasible radii for given centers."""
    r = np.full(N, 0.5)
    # Boundary limits
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise limits
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            r[i] = min(r[i], d / 2.0)
            r[j] = min(r[j], d / 2.0)
            
    return r * shrink_factor

def generate_initial_guesses():
    """Generate diverse, feasible initial configurations."""
    guesses = []
    
    # 1. Scaled Hexagonal Lattice (6-5-6-5-4 rows)
    rows = [6, 5, 6, 5, 4]
    for scale in [0.88, 0.92, 0.96]:
        centers = []
        r_temp = 1.0
        y = r_temp
        for row_idx, count in enumerate(rows):
            x_start = r_temp if row_idx % 2 == 0 else 2 * r_temp
            x = x_start
            for _ in range(count):
                centers.append([x, y])
                x += 2 * r_temp
            y += np.sqrt(3) * r_temp
        centers = np.array(centers)
        
        # Normalize to unit square with margin
        cx_min, cy_min = centers.min(axis=0)
        cx_max, cy_max = centers.max(axis=0)
        centers = (centers - [cx_min, cy_min]) / ([cx_max - cx_min, cy_max - cy_min])
        margin = (1.0 - scale) / 2.0
        centers = centers * scale + margin
        
        r_init = compute_feasible_radii(centers, shrink_factor=0.95)
        guesses.append(np.concatenate([centers[:, 0], centers[:, 1], r_init]))
        
    # 2. Uniform Grid Perturbations
    grid_pts = []
    for i in range(6):
        for j in range(5):
            grid_pts.append([0.05 + i * 0.16, 0.05 + j * 0.2])
    grid_pts = np.array(grid_pts[:N])
    
    for seed in range(6):
        np.random.seed(seed)
        c = grid_pts + np.random.uniform(-0.025, 0.025, grid_pts.shape)
        c = np.clip(c, 0.02, 0.98)
        r_init = compute_feasible_radii(c, shrink_factor=0.92)
        guesses.append(np.concatenate([c[:, 0], c[:, 1], r_init]))
        
    # 3. Random Dense Placements
    for seed in range(6):
        np.random.seed(1000 + seed)
        c = np.random.uniform(0.12, 0.88, (N, 2))
        r_init = compute_feasible_radii(c, shrink_factor=0.9)
        guesses.append(np.concatenate([c[:, 0], c[:, 1], r_init]))
        
    return guesses

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    initial_guesses = generate_initial_guesses()
    
    # Phase 1: Multi-start optimization
    for x0 in initial_guesses:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                # Strict feasibility check
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = initial_guesses[0]
        best_sum = -objective(best_v)
        
    # Phase 2: Iterative perturbation and refinement
    current_v = best_v
    for step in range(50):
        np.random.seed(step * 13 + 7)
        v_pert = current_v.copy()
        
        # Perturb centers to explore configuration space
        v_pert[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Shrink radii to guarantee feasibility after perturbation
        v_pert[2*N:] *= 0.975
        v_pert[2*N:] += np.random.uniform(-0.002, 0.002, N)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.001, 0.45)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = -res.fun
                    best_v = res.x.copy()
                    current_v = best_v
        except Exception:
            pass
            
    # Extract best configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # 2. Enforce non-overlap constraints iteratively
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    # Split overlap equally and add safety margin
                    s = excess / 2.0 + 1e-9
                    radii[i] -= s
                    radii[j] -= s
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
