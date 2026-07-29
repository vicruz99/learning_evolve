# sol_000141 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=a8c275f5 sum of radii=2.628596 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
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
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[PAIR_I] + r[PAIR_J])**2
    c_pair = dist_sq - r_sum_sq
    
    return np.concatenate([c_bound, c_pair])

def ensure_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x, y, r = v[:N].copy(), v[N:2*N].copy(), v[2*N:].copy()
    
    # Enforce boundary constraints
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    
    # Enforce non-overlap constraints iteratively
    for _ in range(5):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        d = np.sqrt(dx**2 + dy**2)
        overlap = (r[PAIR_I] + r[PAIR_J]) - d
        if np.max(overlap) < 1e-9:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return np.concatenate([x, y, r])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    inits = []
    
    np.random.seed(42)
    
    # 1. Staggered hexagonal lattices with varied densities and shifts
    for r0 in np.linspace(0.09, 0.11, 5):
        for shift in np.linspace(-0.02, 0.02, 3):
            for offset_x in [0.0, r0]:
                pts = []
                y = r0 + shift
                row = 0
                while len(pts) < N + 5:
                    x_start = r0 + offset_x
                    x = x_start
                    while x <= 1.0 - r0 and len(pts) < N + 5:
                        pts.append([x, y])
                        x += 2.0 * r0
                    y += np.sqrt(3) * r0
                    row += 1
                pts = np.array(pts[:N])
                pts += np.random.uniform(-0.005, 0.005, pts.shape)
                pts = np.clip(pts, 0.01, 0.99)
                inits.append(np.concatenate([pts[:,0], pts[:,1], np.full(N, 0.04)]))
                
    # 2. Rotated hexagonal lattices to break boundary symmetry
    base_pts = []
    y = 0.1
    row = 0
    while len(base_pts) < N + 10:
        x = 0.1
        while x <= 0.9:
            base_pts.append([x, y])
            x += 0.2
        y += 0.1 * np.sqrt(3)
        row += 1
    base_pts = np.array(base_pts[:N+10])
    
    for angle in np.linspace(-0.15, 0.15, 5):
        c, s = np.cos(angle), np.sin(angle)
        rot = base_pts - 0.5
        rot = rot @ np.array([[c, -s], [s, c]]) + 0.5
        valid = (rot[:,0] > 0.05) & (rot[:,0] < 0.95) & \
                (rot[:,1] > 0.05) & (rot[:,1] < 0.95)
        if np.sum(valid) >= N:
            pts = rot[valid][:N]
            pts += np.random.uniform(-0.005, 0.005, pts.shape)
            pts = np.clip(pts, 0.01, 0.99)
            inits.append(np.concatenate([pts[:,0], pts[:,1], np.full(N, 0.04)]))
            
    # 3. Random placements relaxed with repulsive forces
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(300):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                    if d < 0.2 and d > 1e-5:
                        f = (0.2 - d) * 0.1 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        inits.append(np.concatenate([pts[:,0], pts[:,1], np.full(N, 0.03)]))
        
    # Primary Multi-Start Optimization
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Adaptive Refinement: Escape local minima by shrinking radii and perturbing centers
    if best_v is not None:
        current = best_v.copy()
        for step in range(20):
            pert = current.copy()
            pert[2*N:] *= 0.97  # Shrink to create breathing room
            pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            pert = ensure_feasible(pert)  # Guarantee valid start
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12})
                s = -res.fun
                if s > best_sum and np.min(constraints(res.x)) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    current = best_v.copy()
            except Exception:
                pass
                
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict Post-Processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0 - centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0 - centers[:,1]))
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(10):
        dx = centers[PAIR_I,0] - centers[PAIR_J,0]
        dy = centers[PAIR_I,1] - centers[PAIR_J,1]
        d = np.sqrt(dx**2 + dy**2)
        overlap = (radii[PAIR_I] + radii[PAIR_J]) - d
        if np.max(overlap) < 1e-9:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        radii[PAIR_I] = np.maximum(0.0, radii[PAIR_I] - shrink)
        radii[PAIR_J] = np.maximum(0.0, radii[PAIR_J] - shrink)
        
    return centers, radii, float(np.sum(radii))
