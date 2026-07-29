# sol_000178 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000150 (state 86f9e7dc) state=ae18d684 sum of radii=2.620761 correctness=1.0
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
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    
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

def make_feasible(centers, radii):
    """Iteratively adjust radii to guarantee strict feasibility."""
    r = radii.copy()
    # Enforce boundaries
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Enforce pairwise non-overlap
    for _ in range(15):
        dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
        dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-6
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return r

def generate_starts():
    """Generate a curated set of diverse initial configurations."""
    starts = []
    
    # Curated hexagonal lattice starts with strategic rotations/shifts
    params = [
        (0.090, 0.00, 0.00, 0.00), (0.095, 0.00, 0.00, 0.00), (0.100, 0.00, 0.00, 0.00),
        (0.090, 0.10, 0.00, 0.00), (0.090, -0.10, 0.00, 0.00),
        (0.095, 0.10, 0.00, 0.00), (0.095, -0.10, 0.00, 0.00),
        (0.100, 0.10, 0.00, 0.00), (0.100, -0.10, 0.00, 0.00),
        (0.095, 0.00, 0.02, 0.00), (0.095, 0.00, -0.02, 0.00),
        (0.095, 0.00, 0.00, 0.02), (0.095, 0.00, 0.00, -0.02),
        (0.100, 0.05, 0.01, 0.01), (0.100, -0.05, -0.01, -0.01)
    ]
    
    for r0, angle, sx, sy in params:
        pts = []
        y = r0 + sy
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + sx + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3.0)
            row += 1
            
        pts = np.array(pts[:N])
        if angle != 0.0:
            c, s = np.cos(angle), np.sin(angle)
            pts = pts - 0.5
            pts = np.column_stack([pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c]) + 0.5
            
        pts = np.clip(pts, 0.02, 0.98)
        r = make_feasible(pts, np.full(N, r0 * 0.85))
        starts.append(np.concatenate([pts[:,0], pts[:,1], r]))
        
    # Force-relaxed random starts
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.2 and d > 1e-4:
                        f = (0.2 - d) / d * 0.1
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces
            pts = np.clip(pts, 0.05, 0.95)
            
        r = make_feasible(pts, np.full(N, 0.03))
        starts.append(np.concatenate([pts[:,0], pts[:,1], r]))
        
    return starts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    starts = generate_starts()
    
    # Phase 1: Multi-start optimization
    for v0 in starts:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local Refinement & Escape from Local Minima
    if best_v is not None:
        curr_v = best_v.copy()
        for step in range(40):
            np.random.seed(step * 100 + 7)
            v_p = curr_v.copy()
            
            # Adaptive noise schedule
            noise = 0.006 * (1.0 - step / 40.0)
            v_p[:2*N] += np.random.uniform(-noise, noise, 2*N)
            v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
            
            # Shrink radii to create slack and guarantee feasibility
            centers_p = v_p[:2*N].reshape(N, 2)
            v_p[2*N:] = make_feasible(centers_p, v_p[2*N:] * 0.94)
            
            try:
                res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
                
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
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
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
