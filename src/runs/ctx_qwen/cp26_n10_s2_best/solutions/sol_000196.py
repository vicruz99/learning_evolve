# sol_000196 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000128 (state 9970da69) state=683a2c08 sum of radii=2.620761 correctness=1.0
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
    
    # Boundary constraints: circles inside [0,1]x[0,1]
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
    r = np.full(N, 0.25)
    
    # Distance to square boundaries
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Distance to other centers
    dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
    dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
    dists = np.hypot(dx, dy)
    r_pairs = dists / 2.0
    
    for k in range(len(PAIR_I)):
        i, j = PAIR_I[k], PAIR_J[k]
        val = r_pairs[k]
        if val < r[i]: r[i] = val
        if val < r[j]: r[j] = val
        
    # Keep very close to theoretical max to start optimization near boundaries
    return r * 0.995

def make_strictly_feasible(centers, radii):
    """Iteratively adjust radii to guarantee strict feasibility for the validator."""
    r = radii.copy()
    for _ in range(15):
        # Enforce boundaries
        r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
        r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        
        # Enforce pairwise non-overlap
        dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
        dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        pos_overlap = np.maximum(0.0, overlap)
        if np.max(pos_overlap) < 1e-11:
            break
            
        # Shrink equally with minimal safety margin
        shrink = pos_overlap / 2.0 + 1e-10
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return np.maximum(r, 1e-9)

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Rotated Hexagonal Lattices
    for r0 in [0.095, 0.100, 0.105]:
        for ang in np.linspace(0.0, 0.30, 12):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 10:
                x_start = r0 if row % 2 == 0 else 2.0 * r0
                x = x_start
                while x <= 1.0 - r0 and len(pts) < N + 10:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
                row += 1
                
            pts = np.array(pts[:N])
            
            # Rotate around center
            if ang != 0.0:
                c, s = np.cos(ang), np.sin(ang)
                pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
                
            pts = np.clip(pts, 0.015, 0.985)
            r = compute_feasible_radii(pts)
            inits.append(np.concatenate([pts[:, 0], pts[:, 1], r]))
            
    # 2. Force-Relaxed Random Scatter
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(250):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.24 and d > 1e-5:
                        f = (0.24 - d) / d * 0.4
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.02
            pts = np.clip(pts, 0.05, 0.95)
            
        r = compute_feasible_radii(pts)
        inits.append(np.concatenate([pts[:, 0], pts[:, 1], r]))
        
    return inits

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization
    inits = generate_inits()
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                # Accept if sufficiently feasible
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = inits[0]
        
    # Phase 2: Iterative perturbation refinement to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(50):
            np.random.seed(step + 1000)
            noise_scale = 0.005 * (1.0 - step / 50.0)
            
            v_p = current_v.copy()
            v_p[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
            v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
            
            # Recompute feasible radii to guarantee valid restart
            centers_p = v_p[:2*N].reshape(N, 2)
            v_p[2*N:] = compute_feasible_radii(centers_p) * 0.95
            
            try:
                res = minimize(objective, v_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                s = -res.fun
                if s > best_sum:
                    if np.min(constraints(res.x)) >= -1e-7:
                        best_sum = s
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                continue
                
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack([cx, cy])
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    cr = make_strictly_feasible(centers, cr)
    
    return centers, cr, float(np.sum(cr))
