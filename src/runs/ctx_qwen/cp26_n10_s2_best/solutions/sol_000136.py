# sol_000136 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=92dbd4f4 sum of radii=2.623217 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for numerical stability and smooth gradients."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    # Distance matrix between all centers
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Distance to square boundaries
    wall_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Initialize at 97% of theoretical max to guarantee strict feasibility while staying close to optimum
    r = 0.97 * np.minimum(min_dists / 2.0, wall_dists)
    return np.clip(r, 1e-5, 0.25)

def generate_initial_configs():
    """Generate diverse initial configurations covering different topological basins."""
    configs = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with various rotations, shifts, and densities
    for _ in range(40):
        angle = np.random.uniform(-0.3, 0.3)
        sx, sy = np.random.uniform(-0.1, 0.1, 2)
        r0 = np.random.uniform(0.095, 0.108)
        
        pts = []
        for i in range(-10, 15):
            for j in range(-10, 15):
                x = i * r0 + (j % 2) * r0 * 0.5
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        # Rotate
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        
        # Center and shift
        pts += [0.5 + sx, 0.5 + sy]
        
        # Filter valid points inside square
        mask = (pts[:, 0] > 0.02) & (pts[:, 0] < 0.98) & (pts[:, 1] > 0.02) & (pts[:, 1] < 0.98)
        pts_valid = pts[mask]
        
        if len(pts_valid) >= N:
            idx = np.random.choice(len(pts_valid), N, replace=False)
            configs.append(pts_valid[idx])
            
    # 2. Staggered row patterns (classical 6-5-6-5-4 arrangement)
    for _ in range(25):
        pts = []
        ry = 0.10 + np.random.uniform(-0.01, 0.01)
        counts = [6, 5, 6, 5, 4]
        for row, cnt in enumerate(counts):
            rx = 0.10 + np.random.uniform(-0.01, 0.01) if row % 2 == 0 else 0.18 + np.random.uniform(-0.01, 0.01)
            step_x = 0.16 + np.random.uniform(-0.005, 0.005)
            for _ in range(cnt):
                pts.append([rx, ry])
                rx += step_x
            ry += 0.145 + np.random.uniform(-0.005, 0.005)
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.01, 0.01, pts.shape)
        configs.append(np.clip(pts, 0.02, 0.98))
        
    # 3. Random dense scatter to catch unexpected asymmetric optima
    for _ in range(20):
        configs.append(np.random.uniform(0.15, 0.85, (N, 2)))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.3)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = generate_initial_configs()
    
    # Phase 1: Multi-start exploration
    for centers in inits:
        r_init = get_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            if -res.fun > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative perturbation & refinement to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(40):
            np.random.seed(step + 888)
            pert = current_v.copy()
            
            # Perturb centers slightly
            pert[:2*N] += np.random.uniform(-0.002, 0.002, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            
            # Recompute feasible radii to guarantee restart feasibility
            centers_p = pert[:2*N].reshape(N, 2)
            pert[2*N:] = get_feasible_radii(centers_p) * 0.98
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                
                if -res.fun > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        best_sum = -res.fun
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
