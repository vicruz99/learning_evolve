# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=42bc631b sum of radii=2.631937 correctness=1.0
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
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    r_sum = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - r_sum**2
    return c

def compute_initial_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(r, np.min(dists, axis=1) / 2.0)
    
    return np.clip(r * 0.85, 0.001, 0.5)

def repulsion_init(seed):
    """Generate a well-spaced initial configuration using repulsive forces."""
    np.random.seed(seed)
    pts = np.random.uniform(0.1, 0.9, (N, 2))
    for _ in range(800):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d = np.hypot(dx, dy)
                if d < 0.25 and d > 1e-4:
                    f = 0.05 / d**2
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                    
        mask_l = pts[:, 0] < 0.1
        mask_r = pts[:, 0] > 0.9
        mask_b = pts[:, 1] < 0.1
        mask_t = pts[:, 1] > 0.9
        forces[mask_l, 0] += 0.2
        forces[mask_r, 0] -= 0.2
        forces[mask_b, 1] += 0.2
        forces[mask_t, 1] -= 0.2
        
        pts += forces * 0.05
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    """Optimizes packing of 26 circles in a unit square to maximize sum of radii."""
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Repulsion-based starts
    for s in range(25):
        inits.append(repulsion_init(s))
        
    # 2. Hexagonal lattice variations
    for s in range(25):
        np.random.seed(s + 100)
        r0 = 0.09 + np.random.uniform(-0.01, 0.01)
        shift = np.random.uniform(-0.02, 0.02, 2)
        angle = np.random.uniform(-0.2, 0.2)
        
        pts = []
        y = r0 + shift[1]
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + shift[0] + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3.0)
            row += 1
            
        pts = np.array(pts[:N])
        cx, cy = 0.5, 0.5
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        c_val, s_val = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c_val, -s_val], [s_val, c_val]])
        pts[:, 0] += cx
        pts[:, 1] += cy
        inits.append(np.clip(pts, 0.02, 0.98))
        
    # 3. Perturbed grids
    for s in range(10):
        np.random.seed(s + 200)
        pts = np.array([[0.08 + i * 0.16 + np.random.uniform(-0.01, 0.01), 
                         0.08 + j * 0.18 + np.random.uniform(-0.01, 0.01)] 
                        for i in range(6) for j in range(5)])[:N]
        inits.append(pts)

    # Phase 1: Multi-start exploration
    for centers in inits:
        r_init = compute_initial_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback initialization
    if best_v is None:
        centers_fallback = inits[0]
        r_fallback = compute_initial_radii(centers_fallback)
        best_v = np.concatenate([centers_fallback[:, 0], centers_fallback[:, 1], r_fallback])
        
    # Phase 2: Local refinement to escape shallow local minima
    current_v = best_v
    for step in range(30):
        pert = current_v.copy()
        pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
        pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
        pert[2*N:] *= 0.95
        pert[2*N:] = compute_initial_radii(pert[:2*N].reshape(N, 2)) * 0.95
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract optimal configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        cr[i] = min(cr[i], centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, cr, float(np.sum(cr))
