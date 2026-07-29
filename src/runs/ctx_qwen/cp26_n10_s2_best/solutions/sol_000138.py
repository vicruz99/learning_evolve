# sol_000138 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=ee651493 sum of radii=2.628457 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Pre-allocate constraint array
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

def get_feasible_r(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Distance to square boundaries
    walls = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                       np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Initialize at 70% of theoretical max to guarantee strict feasibility
    r = 0.70 * np.minimum(min_dists / 2.0, walls)
    return np.clip(r, 1e-5, 0.25)

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with various rotations and densities
    for seed in range(15):
        np.random.seed(seed)
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        cx_shift, cy_shift = np.random.uniform(-0.05, 0.05, 2)
        
        pts = []
        r0 = 0.10 + np.random.uniform(-0.01, 0.01)
        for i in range(-5, 10):
            for j in range(-5, 10):
                x = i * r0 + (j % 2) * r0 * 0.5
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        # Rotate
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        
        # Center and shift
        pts -= pts.mean(axis=0)
        pts += [0.5 + cx_shift, 0.5 + cy_shift]
        
        # Filter valid points inside square
        mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) >= N:
            idx = np.random.choice(valid_indices, N, replace=False)
            inits.append(pts[idx])
            
    # 2. Perturbed square grids
    for seed in range(10):
        np.random.seed(seed + 100)
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.1 + i*0.16 + np.random.uniform(-0.02, 0.02), 
                            0.1 + j*0.20 + np.random.uniform(-0.02, 0.02)])
        if len(pts) >= N:
            inits.append(np.array(pts[:N]))

    # 3. Random dense scatter
    for seed in range(10):
        np.random.seed(seed + 200)
        inits.append(np.random.uniform(0.15, 0.85, size=(N, 2)))
        
    # 4. Greedy farthest-point sampling (space-filling)
    for seed in range(8):
        np.random.seed(seed + 300)
        pool = np.random.uniform(0.0, 1.0, (2000, 2))
        pts = [[0.5, 0.5]]
        
        for _ in range(N - 1):
            dists = np.linalg.norm(pool[:, None, :] - np.array(pts)[None, :, :], axis=2)
            min_d = np.min(dists, axis=1)
            walls = np.minimum(np.minimum(pool[:, 0], 1.0 - pool[:, 0]), 
                               np.minimum(pool[:, 1], 1.0 - pool[:, 1]))
            scores = np.minimum(min_d, walls)
            idx = np.argmax(scores)
            pts.append(pool[idx])
            pool = np.delete(pool, idx, axis=0)
        inits.append(np.array(pts))
        
    return inits

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = generate_inits()
    
    # Phase 1: Multi-start optimization
    for centers in inits:
        r_init = get_feasible_r(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            
            if -res.fun > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_v is not None:
        for step in range(40):
            np.random.seed(step + 500)
            v_pert = best_v.copy()
            v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
            
            # Shrink and recompute feasible radii to guarantee restart feasibility
            centers_pert = v_pert[:2*N].reshape(N, 2)
            v_pert[2*N:] = get_feasible_r(centers_pert) * 0.95
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                if -res.fun > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-6:
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                continue
                
    # Fallback (highly unlikely to trigger)
    if best_v is None:
        centers_fb = np.random.uniform(0.2, 0.8, (N, 2))
        r_fb = get_feasible_r(centers_fb)
        best_v = np.concatenate([centers_fb[:, 0], centers_fb[:, 1], r_fb])
        
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(15):
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
