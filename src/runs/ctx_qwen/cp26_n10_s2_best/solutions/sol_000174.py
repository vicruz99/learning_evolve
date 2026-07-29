# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000150 (state 86f9e7dc) state=e84196fd sum of radii=2.625473 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
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
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(r, np.min(dists, axis=1) / 2.0)
    
    return np.clip(r * 0.85, 0.001, 0.25)

def repulsion_init(centers, steps=150):
    """Spread centers using repulsive forces to improve initial packing density."""
    pts = centers.copy()
    n = len(pts)
    target_dist = 0.22
    
    for _ in range(steps):
        diff = pts[:, None, :] - pts[None, :, :]
        dist_sq = np.sum(diff**2, axis=2) + 1e-12
        dist = np.sqrt(dist_sq)
        
        mask = dist < target_dist
        mask[np.eye(n, dtype=bool)] = False
        
        f_mag = np.zeros_like(dist)
        f_mag[mask] = 0.1 * (target_dist - dist[mask])
        f_vec = diff * f_mag[:, :, None] / dist[:, :, None]
        forces = np.sum(f_vec, axis=1)
        
        pts += forces * 0.05
        pts = np.clip(pts, 0.04, 0.96)
        
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    inits = []
    
    # Phase 1: Generate diverse initial configurations
    # 1. Rotated Hexagonal Lattices with various densities and shifts
    for r0 in [0.09, 0.10]:
        for angle in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for sx in [-0.02, 0.0, 0.02]:
                for sy in [-0.02, 0.0, 0.02]:
                    pts = []
                    y = r0 + sy
                    row = 0
                    while len(pts) < N + 10:
                        x_start = r0 + sx + (row % 2) * r0
                        x = x_start
                        while x <= 1.0 - r0 and len(pts) < N + 10:
                            pts.append([x, y])
                            x += 2.0 * r0
                        y += r0 * np.sqrt(3.0)
                        row += 1
                        
                    pts = np.array(pts[:N])
                    if angle != 0.0:
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        pts = (pts - 0.5) @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]) + 0.5
                        
                    pts = repulsion_init(pts, steps=150)
                    r_init = compute_feasible_radii(pts)
                    inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
                    
    # 2. Force-relaxed random starts
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        pts = repulsion_init(pts, steps=200)
        r_init = compute_feasible_radii(pts)
        inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # Primary Optimization Pass
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative Refinement & Escape from Local Minima
    if best_v is not None:
        curr_v = best_v.copy()
        for step in range(25):
            np.random.seed(step + 500)
            pert = curr_v.copy()
            
            # Decaying noise for centers
            noise = 0.004 * (1.0 - step / 25.0)
            pert[:2*N] += np.random.uniform(-noise, noise, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            
            # Aggressive shrinkage to break rigid contact networks
            shrink = 0.85 - step * 0.005
            pert[2*N:] *= max(0.7, shrink)
            
            # Recompute strictly feasible radii for perturbed centers
            centers_p = pert[:2*N].reshape(N, 2)
            r_p = compute_feasible_radii(centers_p)
            pert[2*N:] = r_p
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
                
                s = -res.fun
                if s > best_sum:
                    cv = constraints(res.x)
                    if np.min(cv) >= -1e-6:
                        best_sum = s
                        best_v = res.x.copy()
                        curr_v = best_v.copy()
            except Exception:
                continue
                
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
