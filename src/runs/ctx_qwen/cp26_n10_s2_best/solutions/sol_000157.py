# sol_000157 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=8fd8a9ec sum of radii=2.628596 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: Minimize negative sum of radii (maximize sum)."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = np.hypot(dx, dy) - (r[PAIR_I] + r[PAIR_J])
    
    return c

def make_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    # Enforce boundary constraints strictly
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    
    # Enforce non-overlap constraints iteratively (vectorized)
    for _ in range(15):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        r[PAIR_I] -= shrink
        r[PAIR_J] -= shrink
        r = np.maximum(r, 0.0)
        
    return np.concatenate([x, y, r])

def get_init_centers(method, seed=None):
    """Generates diverse initial center configurations."""
    if seed is not None:
        np.random.seed(seed)
        
    if method == 'repel':
        # Force-directed layout to evenly spread points
        pts = np.random.uniform(0.2, 0.8, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.25 and d > 1e-4:
                        f = (0.25 - d) * 0.6 / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.025
            pts = np.clip(pts, 0.05, 0.95)
        return pts
        
    elif method == 'hex':
        # Hexagonal lattice with random rotation and shift
        r0 = 0.090 + np.random.uniform(-0.010, 0.015)
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 8:
            x_start = r0 if row % 2 == 0 else 2 * r0
            x = x_start
            while x <= 1 - r0 and len(pts) < N + 8:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
            
        pts = np.array(pts[:N])
        angle = np.random.uniform(-0.25, 0.25)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        pts = pts - 0.5
        pts = pts @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]) + 0.5
        return pts
        
    else: # random
        return np.random.uniform(0.15, 0.85, (N, 2))

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.25)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Generate diverse initial configurations
    configs = []
    for seed in range(25):
        configs.append(get_init_centers('hex', seed))
    for seed in range(10):
        configs.append(get_init_centers('repel', seed))
    for seed in range(5):
        configs.append(get_init_centers('rand', seed))
        
    # Phase 1: Multi-start optimization
    for centers in configs:
        # Compute tight feasible initial radii
        r_init = np.full(N, 0.5)
        for i in range(N):
            r_init[i] = min(centers[i,0], 1.0 - centers[i,0], 
                            centers[i,1], 1.0 - centers[i,1])
            for j in range(N):
                if i != j:
                    d = np.hypot(centers[i,0] - centers[j,0], 
                                 centers[i,1] - centers[j,1])
                    if d / 2.0 < r_init[i]:
                        r_init[i] = d / 2.0
        r_init *= 0.88  # Leave slack for optimizer
        
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, 
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if completely failed (unlikely)
    if best_v is None:
        best_v = make_feasible(np.concatenate([
            np.random.uniform(0.2, 0.8, (N, 2)).flatten(), 
            np.full(N, 0.05)
        ]))
        
    # Phase 2: Shake & Refine to escape local minima
    current_v = best_v.copy()
    for step in range(20):
        pert = current_v.copy()
        
        # Shrink radii to unstick circles from local jamming
        shrink_factor = 0.94 - step * 0.002
        pert[2*N:] *= max(0.85, shrink_factor)
        
        # Perturb centers to explore new basins
        pert[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
        pert[:2*N] = np.clip(pert[:2*N], 0.02, 0.98)
        
        # Guarantee strict feasibility before optimization
        pert = make_feasible(pert)
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': 3500, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage and safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
