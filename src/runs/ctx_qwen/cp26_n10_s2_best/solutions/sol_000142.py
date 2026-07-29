# sol_000142 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=b71d7332 sum of radii=2.634292 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles must be inside [0, 1]x[0, 1]
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints (squared distance for stability)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dist_sq - r_sum**2
    
    return np.concatenate([c_bound, c_pair])

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.full(N, 0.25)
    
    # Distance to square boundaries
    wall_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    r = np.minimum(r, wall_dists)
    
    # Distance to other centers
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    r = np.minimum(r, min_dists / 2.0)
    
    # Scale down slightly to guarantee strict feasibility for optimizer
    return np.clip(r * 0.88, 1e-4, 0.25)

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with various rotations and densities
    for seed in range(12):
        np.random.seed(seed)
        r0 = 0.10 + np.random.uniform(-0.02, 0.02)
        angle = np.random.uniform(-0.25, 0.25)
        
        pts = []
        y = r0 + np.random.uniform(-0.02, 0.02)
        row = 0
        while len(pts) < N + 8:
            x_start = r0 + (row % 2) * r0 + np.random.uniform(-0.01, 0.01)
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 8:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
            
        pts = np.array(pts[:N])
        
        # Rotate around center
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        pts = np.clip(pts, 0.04, 0.96)
        
        r = get_feasible_radii(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r]))
        
    # 2. Perturbed square grids
    for seed in range(8):
        np.random.seed(seed + 100)
        pts = np.array([[0.12 + i*0.15 + np.random.uniform(-0.02, 0.02), 
                         0.12 + j*0.18 + np.random.uniform(-0.02, 0.02)]
                        for i in range(6) for j in range(5)][:N])
        pts = np.clip(pts, 0.04, 0.96)
        r = get_feasible_radii(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r]))

    # 3. Force-relaxed random scatter
    for seed in range(8):
        np.random.seed(seed + 200)
        pts = np.random.uniform(0.15, 0.85, size=(N, 2))
        # Quick repulsion relaxation
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.22 and d > 1e-4:
                        f = (0.22 - d) / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.015
            pts = np.clip(pts, 0.04, 0.96)
            
        r = get_feasible_radii(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r]))
        
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
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_val = constraints(res.x)
                # Accept if sufficiently feasible
                if np.min(c_val) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_v is None:
        best_v = inits[0]
        best_sum = -np.sum(best_v[2*N:])
        
    # Phase 2: Local perturbation refinement to escape shallow local minima
    current_v = best_v.copy()
    for step in range(15):
        np.random.seed(step + 500)
        v_pert = current_v.copy()
        
        # Perturb centers slightly
        v_pert[:2*N] += np.random.uniform(-0.003, 0.003, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
        
        # Shrink radii and recompute feasible radii to guarantee restart feasibility
        centers_pert = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = get_feasible_radii(centers_pert) * 0.92
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal shrinkage
    # Validator allows: dist >= r1 + r2 - 1e-12
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                sum_r = cr[i] + cr[j]
                # Add tiny buffer to ensure strict compliance after floating point ops
                if sum_r > d + 1e-11:
                    shrink = (sum_r - d - 1e-11) / 2.0
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, cr, float(np.sum(cr))
