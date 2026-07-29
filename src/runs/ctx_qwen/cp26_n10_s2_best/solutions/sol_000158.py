# sol_000158 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=a03b12ea sum of radii=2.630179 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and non-overlap (squared distances)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c_bound = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints (squared for smooth gradients)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dx**2 + dy**2 - r_sum**2
    
    return np.concatenate([c_bound, c_pair])

def compute_init_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        # Distance to walls
        r[i] = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        # Distance to other centers
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            val = d / 2.0
            if val < r[i]: r[i] = val
            if val < r[j]: r[j] = val
    # Shrink slightly to guarantee strict feasibility for optimizer start
    return r * 0.85

def force_relax(centers, steps=400):
    """Relax configuration using repulsive forces to find dense local packing."""
    pts = centers.copy()
    n = len(pts)
    
    for _ in range(steps):
        # Vectorized pairwise force calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-12)
        
        # Mask for active repulsion range and exclude self-interaction
        mask = (dist < 0.25) & (np.eye(n, dtype=bool) == False)
        
        f_mag = np.zeros_like(dist)
        f_mag[mask] = (0.25 - dist[mask]) * 0.5
        f_vec = diff * f_mag[:, :, np.newaxis] / (dist[:, :, np.newaxis] + 1e-12)
        
        forces = np.sum(f_vec, axis=1)
        
        pts += forces * 0.05
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = []
    np.random.seed(42)
    
    # 1. Force-relaxed random starts
    for seed in range(15):
        np.random.seed(seed)
        c = np.random.uniform(0.15, 0.85, (N, 2))
        c = force_relax(c, steps=500)
        r = compute_init_radii(c)
        inits.append(np.concatenate([c[:,0], c[:,1], r]))
        
    # 2. Rotated hexagonal lattice starts
    for seed in range(15):
        np.random.seed(seed + 1000)
        r0 = 0.095 + np.random.uniform(-0.01, 0.01)
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        
        # Apply random rotation and shift
        angle = np.random.uniform(-0.3, 0.3)
        shift = np.random.uniform(-0.05, 0.05, 2)
        c, s = np.cos(angle), np.sin(angle)
        pts = pts - 0.5
        pts = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c]) + 0.5 + shift
        pts = np.clip(pts, 0.02, 0.98)
        
        r = compute_init_radii(pts)
        inits.append(np.concatenate([pts[:,0], pts[:,1], r]))
        
    # Primary multi-start optimization
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum and np.min(constraints(res.x)) >= -1e-7:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Adaptive refinement to escape local minima
    if best_v is not None:
        curr_v = best_v
        for step in range(20):
            np.random.seed(step + 2000)
            pert = curr_v.copy()
            
            # Perturb centers
            pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            
            # Shrink radii to create breathing room, then re-feasibilize
            c_pts = pert[:2*N].reshape(N, 2)
            r_pts = compute_init_radii(c_pts)
            pert[2*N:] = r_pts * 0.85
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                s = -res.fun
                if s > best_sum and np.min(constraints(res.x)) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    curr_v = best_v.copy()
            except Exception:
                pass
                
    # Fallback
    if best_v is None:
        best_v = inits[0]
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing for validator compliance
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        radii[i] = max(radii[i], 0.0)
        
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d + 1e-10:
                    excess = radii[i] + radii[j] - d
                    radii[i] -= excess * 0.5
                    radii[j] -= excess * 0.5
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
