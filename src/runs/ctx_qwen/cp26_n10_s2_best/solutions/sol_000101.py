# sol_000101 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state af044a19) state=7d05ddd0 sum of radii=2.614067 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for vectorized constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for better numerical conditioning."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles must be inside [0,1]x[0,1]
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
    r_sum_sq = (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c_bound, dist_sq - r_sum_sq])

def generate_starts():
    """Generates a diverse set of initial configurations for multi-start optimization."""
    starts = []
    
    # 1. Hexagonal lattices with various base radii, shifts, and rotations
    for r0 in [0.08, 0.09, 0.10, 0.11]:
        for shift_y in [0.0, 0.01, -0.01]:
            for rot in [0.0, 0.15, 0.3, np.pi/12]:
                pts = []
                y = r0 + shift_y
                row = 0
                while len(pts) < N + 10:
                    x_start = r0 + (row % 2) * r0
                    x = x_start
                    while x <= 1 - r0:
                        pts.append([x, y])
                        x += 2 * r0
                    y += np.sqrt(3) * r0
                    row += 1
                pts = np.array(pts[:N])
                
                if rot != 0.0:
                    c = pts - 0.5
                    c = np.dot(c, [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                    pts = c + 0.5
                    
                pts = np.clip(pts, 0.02, 0.98)
                r_init = np.full(N, 0.05)
                starts.append(np.concatenate([pts.flatten(), r_init]))
                
    # 2. Square grid configurations with shifts
    for shift in np.linspace(0, 0.12, 5):
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([i*0.17 + 0.09 + shift, j*0.17 + 0.09 + shift])
        pts = np.array(pts[:N])
        r_init = np.full(N, 0.05)
        starts.append(np.concatenate([pts.flatten(), r_init]))
        
    # 3. Purely random configurations
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        r_init = np.random.uniform(0.03, 0.06, N)
        starts.append(np.concatenate([pts.flatten(), r_init]))
        
    # 4. Corner/edge biased configurations (often optimal for variable radii)
    for seed in range(8):
        np.random.seed(200 + seed)
        pts = np.vstack([
            [0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85],
            np.random.uniform(0.2, 0.8, (N-4, 2))
        ])
        r_init = np.random.uniform(0.03, 0.07, N)
        starts.append(np.concatenate([pts.flatten(), r_init]))
        
    return starts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    starts = generate_starts()
    
    # Phase 1: Multi-start exploration
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict, 
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                # Strict feasibility check
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-6:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement around the best solution to escape local minima
    if best_v is not None:
        for seed in range(12):
            np.random.seed(seed + 500)
            x0 = best_v.copy()
            
            # Perturb centers to explore nearby basins
            x0[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
            x0[:2*N] = np.clip(x0[:2*N], 0.01, 0.99)
            
            # Shrink radii slightly to guarantee a feasible starting point
            x0[2*N:] *= 0.94
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                               
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-6:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                continue
                
    # Fallback (should not be reached)
    if best_v is None:
        best_v = starts[0]
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        max_r_bound = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        if radii[i] > max_r_bound:
            radii[i] = max_r_bound
            
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if radii[i] + radii[j] > d:
                    excess = radii[i] + radii[j] - d
                    shrink = excess / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
