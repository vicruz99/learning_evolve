# sol_000137 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=34f75770 sum of radii=2.607347 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: Minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Pre-allocate constraint array for performance
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def compute_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    # Vectorized distance matrix
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Distance to square boundaries
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Initialize at 92% of theoretical max to guarantee strict feasibility slack
    return np.clip(0.92 * np.minimum(min_dists / 2.0, wall_dists), 1e-4, 0.25)

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Rotated and shifted hexagonal lattices
    for seed in range(12):
        np.random.seed(seed)
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        c, s = np.cos(angle), np.sin(angle)
        
        pts = []
        r0 = 0.095 + np.random.uniform(-0.01, 0.01)
        for i in range(-6, 8):
            for j in range(-6, 8):
                x = i * r0 + (j % 2) * 0.5 * r0
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        # Rotate and center
        pts = pts @ np.array([[c, -s], [s, c]])
        pts -= pts.mean(axis=0)
        pts += np.random.uniform(0.4, 0.6, 2)
        
        mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & \
               (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
        valid = pts[mask]
        if len(valid) >= N:
            idx = np.random.choice(len(valid), N, replace=False)
            configs.append(valid[idx])
            
    # 2. Dense random scatterings
    for seed in range(15):
        np.random.seed(seed + 100)
        configs.append(np.random.uniform(0.15, 0.85, size=(N, 2)))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.25)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for centers in inits:
        r_init = compute_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Verify strict feasibility with tolerance
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative perturbation refinement to escape local minima
    if best_v is not None:
        for step in range(50):
            np.random.seed(step + 2000)
            # Gradually shrink radii to create breathing room for center rearrangement
            scale = 0.985 - step * 0.0008
            v_pert = best_v.copy()
            v_pert[2*N:] *= scale
            
            # Gaussian perturbation of centers
            v_pert[:2*N] += np.random.normal(0, 0.002, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            
            # Recompute feasible radii for perturbed centers
            centers_pert = v_pert[:2*N].reshape(N, 2)
            v_pert[2*N:] = compute_feasible_radii(centers_pert)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    if np.min(constraints(res.x)) >= -1e-8:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                continue
                
    # Fallback initialization (highly unlikely to trigger)
    if best_v is None:
        centers_fallback = np.random.uniform(0.2, 0.8, (N, 2))
        radii_fallback = compute_feasible_radii(centers_fallback)
        best_v = np.concatenate([centers_fallback[:, 0], centers_fallback[:, 1], radii_fallback])
        
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
