# sol_000146 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=deb4f8d9 sum of radii=2.628434 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for efficient constraint evaluation
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
    
    # Boundary constraints
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def compute_feasible_radii(centers, shrink_factor=0.85):
    """Compute strictly feasible initial radii based on local geometry."""
    # Distance matrix between all centers
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Distance to square boundaries
    wall_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Initialize at shrink_factor of theoretical max to guarantee strict feasibility
    r = shrink_factor * np.minimum(min_dists / 2.0, wall_dists)
    return np.clip(r, 0.01, 0.25)

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with various row patterns and rotations
    row_patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], 
        [4,6,6,6,4], [7,5,5,5,4], [6,6,5,5,4],
        [6,5,5,5,5], [5,5,5,5,6]
    ]
    
    for pat in row_patterns:
        for r0 in [0.09, 0.095, 0.10]:
            pts = []
            y = r0
            row_idx = 0
            for count in pat:
                x_start = r0 if row_idx % 2 == 0 else 2.0 * r0
                for k in range(count):
                    x = x_start + k * 2.0 * r0
                    if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                        pts.append([x, y])
                y += np.sqrt(3) * r0
                row_idx += 1
                
            if len(pts) >= N:
                pts_arr = np.array(pts)
                np.random.shuffle(pts_arr)
                configs.append(pts_arr[:N])
                
                # Try rotated versions
                for ang in [np.pi/12, -np.pi/12, np.pi/6]:
                    c, s = np.cos(ang), np.sin(ang)
                    rot_pts = (pts_arr[:N] - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
                    mask = (rot_pts[:, 0] > 0.05) & (rot_pts[:, 0] < 0.95) & \
                           (rot_pts[:, 1] > 0.05) & (rot_pts[:, 1] < 0.95)
                    if np.sum(mask) >= N:
                        configs.append(rot_pts[mask][:N])

    # 2. Perturbed square grids
    for seed in range(10):
        np.random.seed(seed + 300)
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08 + i*0.16 + np.random.uniform(-0.015, 0.015), 
                            0.08 + j*0.18 + np.random.uniform(-0.015, 0.015)])
        if len(pts) >= N:
            configs.append(np.array(pts[:N]))

    # 3. Random dense scatter with repulsion relaxation
    for seed in range(15):
        np.random.seed(seed + 400)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Simple repulsion relaxation to spread points
        for _ in range(200):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.15 and d > 1e-4:
                        f = (0.15 - d) / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.02
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
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
        r_init = compute_feasible_radii(centers, shrink_factor=0.80)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            
            if -res.fun > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Homotopy Growth Refinement
    # Iteratively scale up radii and re-optimize centers to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(15):
            # Scale radii up slightly
            current_v[2*N:] *= 1.008
            
            # Perturb centers slightly to help find new equilibrium
            current_v[:2*N] += np.random.uniform(-0.002, 0.002, 2*N)
            current_v[:2*N] = np.clip(current_v[:2*N], 0.02, 0.98)
            
            # Ensure feasibility before optimization
            current_v[2*N:] *= 0.95 # Slight shrink to guarantee feasibility start
            
            try:
                res = minimize(objective, current_v, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                
                if -res.fun > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                continue
                
    # Phase 3: Local perturbation refinement
    if best_v is not None:
        for step in range(10):
            np.random.seed(step + 600)
            v_pert = best_v.copy()
            v_pert[:2*N] += np.random.uniform(-0.003, 0.003, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
            v_pert[2*N:] *= 0.96
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                continue
                
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 4: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
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
