# sol_000172 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000150 (state 86f9e7dc) state=a2bd9342 sum of radii=2.631094 correctness=1.0
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

def compute_max_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
    dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
    dists = np.hypot(dx, dy)
    r_pairs = dists / 2.0
    
    for k in range(len(PAIR_I)):
        i, j = PAIR_I[k], PAIR_J[k]
        val = r_pairs[k]
        if val < r[i]: r[i] = val
        if val < r[j]: r[j] = val
        
    return r * 0.85

def make_feasible(centers, r):
    """Iteratively adjust radii to guarantee strict feasibility."""
    r = r.copy()
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    for _ in range(15):
        dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
        dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) * 0.5 + 1e-7
        r[PAIR_I] -= shrink
        r[PAIR_J] -= shrink
        
    return np.maximum(r, 1e-6)

def force_relax(pts, iters=200):
    """Relax configuration using repulsive forces to find dense local packing."""
    pts = pts.copy()
    for _ in range(iters):
        forces = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        mask = dists < 0.2
        f_mag = np.zeros_like(dists)
        f_mag[mask] = (0.2 - dists[mask]) / dists[mask]
        f_vec = diff * f_mag[:, :, np.newaxis]
        forces = np.sum(f_vec, axis=1)
        
        pts += forces * 0.05
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with variations
    for r0 in [0.085, 0.095, 0.105]:
        for angle in np.linspace(-0.2, 0.2, 5):
            for sx in [-0.02, 0.0, 0.02]:
                for sy in [-0.02, 0.0, 0.02]:
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
                        pts = pts @ np.array([[c, -s], [s, c]]) + 0.5
                        
                    pts = np.clip(pts, 0.02, 0.98)
                    r_init = compute_max_radii(pts)
                    inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
                    
    # 2. Force-relaxed random starts
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        pts = force_relax(pts, iters=200)
        r_init = compute_max_radii(pts)
        inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # 3. Staggered grid starts
    for s in np.linspace(0.0, 0.05, 4):
        pts = np.array([[0.08 + i*0.17 + s, 0.08 + j*0.20 + s] 
                        for i in range(6) for j in range(5)])[:N]
        pts = np.clip(pts, 0.02, 0.98)
        r_init = compute_max_radii(pts)
        inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    return inits

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
    
    inits = generate_inits()
    
    # Phase 1: Multi-start optimization
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-5:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = inits[0]
        
    # Phase 2: Iterative refinement to escape local minima
    curr_v = best_v.copy()
    for step in range(25):
        np.random.seed(step + 50)
        v_p = curr_v.copy()
        
        # Perturb centers with decaying noise
        noise = 0.004 * (1.0 - step / 25.0)
        v_p[:2*N] += np.random.uniform(-noise, noise, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Shrink radii to create slack and guarantee feasibility
        centers_p = v_p[:2*N].reshape(N, 2)
        r_p = make_feasible(centers_p, v_p[2*N:] * 0.92)
        v_p[2*N:] = r_p
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-5:
                    best_sum = s
                    best_v = res.x.copy()
                    curr_v = best_v.copy()
        except Exception:
            pass
            
    # Extract final configuration
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
