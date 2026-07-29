# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=62449400 sum of radii=2.625930 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Using squared distances avoids sqrt singularities and provides smoother gradients
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c_pair = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c_bound, c_pair])

def compute_feasible_radii(centers, scale=0.85):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.full(N, 0.5)
    for i in range(N):
        # Distance to boundaries
        dist_bound = min(centers[i, 0], 1.0 - centers[i, 0], 
                         centers[i, 1], 1.0 - centers[i, 1])
        r[i] = dist_bound
        
        # Distance to other centers
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            r[i] = min(r[i], d / 2.0)
            
    return r * scale

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with rotations and shifts
    for seed in range(20):
        r0 = 0.08 + np.random.uniform(-0.01, 0.02)
        angle = np.random.uniform(-0.3, 0.3)
        shift = np.random.uniform(-0.03, 0.03, 2)
        
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
            
        pts = np.array(pts)
        # Rotate around center
        pts = pts - 0.5
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, -s], [s, c]])
        pts = pts + 0.5 + shift
        
        mask = (pts[:, 0] > 0.02) & (pts[:, 0] < 0.98) & \
               (pts[:, 1] > 0.02) & (pts[:, 1] < 0.98)
        pts_valid = pts[mask]
        
        if len(pts_valid) >= N:
            idx = np.random.choice(len(pts_valid), N, replace=False)
            centers = pts_valid[idx]
            centers += np.random.uniform(-0.005, 0.005, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            r_init = compute_feasible_radii(centers)
            configs.append(np.concatenate([centers[:, 0], centers[:, 1], r_init]))
            
    # 2. Perturbed Grids
    for seed in range(10):
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.1 + i*0.16 + np.random.uniform(-0.02, 0.02), 
                            0.1 + j*0.20 + np.random.uniform(-0.02, 0.02)])
        pts = np.array(pts[:N])
        pts = np.clip(pts, 0.02, 0.98)
        r_init = compute_feasible_radii(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # 3. Force-directed repulsion starts
    for seed in range(15):
        np.random.seed(seed + 1000)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        # Quick repulsion relaxation to spread points
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.25 and d > 1e-4:
                        f = (0.25 - d) * 0.5 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.02
            pts = np.clip(pts, 0.05, 0.95)
        r_init = compute_feasible_radii(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.45)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    initial_configs = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for v0 in initial_configs:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback
    if best_v is None:
        best_v = initial_configs[0]
        
    # Phase 2: Local escape & refinement to escape shallow local minima
    current_v = best_v.copy()
    for step in range(25):
        # Shrink radii progressively to unstick circles from boundaries/neighbors
        shrink_factor = 0.90 - step * 0.004
        perturbed = current_v.copy()
        perturbed[2*N:] *= max(0.75, shrink_factor)
        
        # Perturb centers to explore new basins of attraction
        noise = np.random.uniform(-0.006, 0.006, 2*N)
        perturbed[:2*N] += noise
        perturbed[:2*N] = np.clip(perturbed[:2*N], 0.01, 0.99)
        
        try:
            res = minimize(objective, perturbed, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
                current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing for validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
