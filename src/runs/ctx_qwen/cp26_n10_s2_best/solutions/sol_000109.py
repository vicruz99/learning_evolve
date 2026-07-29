# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000055 (state f6ce444f) state=7dacefa3 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to avoid repeated generation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for better numerical stability.
    Returns array where all elements >= 0 indicates feasibility.
    """
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
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum = r[PAIR_I] + r[PAIR_J]
    
    return np.concatenate([c_bound, dist_sq - r_sum**2])

def get_feasible_init(v_centers):
    """
    Given initial centers, compute strictly feasible radii (75% of max possible)
    to provide a strong starting point for the optimizer.
    """
    r_safe = np.full(N, 0.5)
    for i in range(N):
        # Distance to walls
        mr = min(v_centers[i, 0], 1.0 - v_centers[i, 0], 
                 v_centers[i, 1], 1.0 - v_centers[i, 1])
        r_safe[i] = mr
        
        # Distance to other circles
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(v_centers[i, 0] - v_centers[j, 0], 
                         v_centers[i, 1] - v_centers[j, 1])
            if d < 2 * r_safe[i]:
                r_safe[i] = d / 2.0
                
    return r_safe * 0.75

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # --- Stage 1: Generate Diverse Initial Configurations ---
    starts = []
    
    # 1. Hexagonal lattices with varying base radii and shifts
    hex_params = [
        (0.09, 0.0, 0.0), (0.095, 0.02, 0.0), (0.10, -0.02, 0.03),
        (0.105, 0.0, -0.02), (0.09, 0.05, 0.05), (0.098, 0.0, 0.0)
    ]
    
    for r0, sx, sy in hex_params:
        for seed in range(4):
            np.random.seed(seed)
            pts = []
            y = r0 + sy
            row = 0
            while len(pts) < N + 10:
                x_start = r0 + sx + (row % 2) * r0
                x = x_start
                while x <= 1 - r0 and len(pts) < N + 10:
                    pts.append([x, y])
                    x += 2 * r0
                y += np.sqrt(3) * r0
                row += 1
                
            pts = np.array(pts[:N])
            pts += np.random.uniform(-0.008, 0.008, pts.shape)
            pts = np.clip(pts, 0.02, 0.98)
            
            r_init = get_feasible_init(pts)
            starts.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
            
    # 2. Perturbed Grid Configurations
    for seed in range(6):
        np.random.seed(50 + seed)
        grid_x = np.linspace(0.1, 0.9, 6)
        grid_y = np.linspace(0.1, 0.9, 5)
        pts = np.array([[x, y] for x in grid_x for y in grid_y])
        # Pick N points with jitter
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx] + np.random.uniform(-0.02, 0.02, (N, 2))
        pts = np.clip(pts, 0.05, 0.95)
        r_init = get_feasible_init(pts)
        starts.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # 3. Random Dense Configurations
    for seed in range(5):
        np.random.seed(200 + seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        r_init = get_feasible_init(pts)
        starts.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))

    # --- Stage 2: Multi-start Optimization ---
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Verify strict feasibility before accepting
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # --- Stage 3: Adaptive Iterative Refinement ---
    if best_v is not None:
        pert_mag = 0.006
        for step in range(25):
            np.random.seed(900 + step)
            
            v_pert = best_v.copy()
            # Perturb centers
            v_pert[:2*N] += np.random.uniform(-pert_mag, pert_mag, 2*N)
            # Slightly shrink radii to guarantee feasibility after perturbation
            v_pert[2*N:] *= 0.997
            
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.005, 0.995)
            v_pert[2*N:] = np.clip(v_pert[2*N:], 0.01, 0.5)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                pass
                
            # Anneal perturbation magnitude
            pert_mag *= 0.94

    # Fallback (highly unlikely)
    if best_v is None:
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        radii = np.full(N, 0.03)
        return centers, radii, float(np.sum(radii))
        
    # --- Stage 4: Extract & Strict Post-Processing ---
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-11:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
