# sol_000185 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000172 (state a2bd9342) state=5cb7b8c5 sum of radii=2.621899 correctness=1.0
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
        
    return r * 0.82

def make_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:].copy()
    
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    
    for _ in range(10):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) * 0.5 + 1e-8
        r[PAIR_I] -= shrink
        r[PAIR_J] -= shrink
        
    return np.concatenate([x, y, np.maximum(r, 1e-6)])

def hex_pattern(row_counts, r0=0.1, shift_x=0.0, shift_y=0.0, angle=0.0):
    """Generates a hexagonal lattice configuration based on row counts."""
    pts = []
    y = r0 + shift_y
    for row_idx, count in enumerate(row_counts):
        x_start = r0 + shift_x + (row_idx % 2) * r0
        for k in range(count):
            pts.append([x_start + k * 2 * r0, y])
        y += r0 * np.sqrt(3)
    pts = np.array(pts)
    
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    return pts

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 5, 5, 5, 4],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 5, 6, 5, 5],
        [8, 5, 5, 5, 3], [6, 4, 6, 6, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [4, 5, 6, 6, 5], [5, 6, 6, 5, 4]
    ]
    
    for pat in patterns:
        if sum(pat) != N:
            continue
        for angle in np.linspace(-0.15, 0.15, 5):
            for sx in [-0.01, 0.0, 0.01]:
                for sy in [-0.01, 0.0, 0.01]:
                    pts = hex_pattern(pat, r0=0.10, shift_x=sx, shift_y=sy, angle=angle)
                    if np.all(pts >= 0.02) and np.all(pts <= 0.98):
                        pts += np.random.uniform(-0.005, 0.005, pts.shape)
                        pts = np.clip(pts, 0.02, 0.98)
                        r_init = compute_max_radii(pts)
                        inits.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
                        
    # Add some force-relaxed random starts
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Quick repulsion
        for _ in range(100):
            forces = np.zeros_like(pts)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            mask = dists < 0.2
            f_mag = np.zeros_like(dists)
            f_mag[mask] = (0.2 - dists[mask]) / dists[mask]
            f_vec = diff * f_mag[:, :, np.newaxis]
            forces = np.sum(f_vec, axis=1)
            pts += forces * 0.03
            pts = np.clip(pts, 0.05, 0.95)
            
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
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = inits[0]
        
    # Phase 2: Iterative refinement to escape local minima
    curr_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 100)
        v_p = curr_v.copy()
        
        # Perturb centers with decaying noise
        noise = 0.005 * (1.0 - step / 40.0)
        v_p[:2*N] += np.random.uniform(-noise, noise, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Shrink radii to create slack and guarantee feasibility
        centers_p = v_p[:2*N].reshape(N, 2)
        r_p = make_feasible(v_p)[2*N:] * 0.90
        v_p[2*N:] = r_p
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
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
