# sol_000164 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=b3b6f498 sum of radii=2.620761 correctness=1.0
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
    """Inequality constraints: boundaries and non-overlap (linear in radii)."""
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
    c[4*N:] = np.sqrt(dx**2 + dy**2) - (r[PAIR_I] + r[PAIR_J])
    return c

def compute_feasible_radii(centers):
    """Compute a strictly feasible initial radius array for given centers."""
    r = np.full(N, 0.5)
    # Distance to walls
    for i in range(N):
        r[i] = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
    # Distance to neighbors
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            if d < 2.0 * r[i]:
                r[i] = d / 2.0
            if d < 2.0 * r[j]:
                r[j] = d / 2.0
    # Safety margin to guarantee strict feasibility for optimizer
    return r * 0.85

def generate_initial_configs():
    """Generate diverse starting configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with rotation and shift
    for r0 in [0.08, 0.09, 0.10, 0.11]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 10:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0 and len(pts) < N + 10:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
        pts = np.array(pts[:N])
        
        angle = np.random.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]) + 0.5
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.01, 0.99)
        configs.append(pts)
        
    # 2. Square grids with jitter
    for sp in [0.15, 0.18, 0.20, 0.22]:
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.05 + i * sp, 0.05 + j * sp])
        np.random.shuffle(pts)
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.01, 0.01, pts.shape)
        pts = np.clip(pts, 0.01, 0.99)
        configs.append(pts)
        
    # 3. Repelled random placements (prevents initial overlaps, gives optimizer headroom)
    for _ in range(15):
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(150):
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.18 and d > 1e-6:
                        f = (0.18 - d) / d * 0.005
                        diff = pts[i] - pts[j]
                        pts[i] += f * diff
                        pts[j] -= f * diff
                pts = np.clip(pts, 0.01, 0.99)
        configs.append(pts)
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    configs = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for cfg in configs:
        r_init = compute_feasible_radii(cfg)
        v0 = np.concatenate([cfg[:, 0], cfg[:, 1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        # Fallback
        fallback_cfg = np.random.uniform(0.15, 0.85, (N, 2))
        r_fall = compute_feasible_radii(fallback_cfg)
        best_v = np.concatenate([fallback_cfg[:, 0], fallback_cfg[:, 1], r_fall])
        best_sum = -np.sum(best_v[2*N:])
        
    # Phase 2: Perturbation & Refinement to escape local minima
    current_v = best_v.copy()
    for step in range(25):
        noise_mag = 0.006 * max(0.0, 1.0 - step / 25.0)
        v_pert = current_v.copy()
        v_pert[:2*N] += np.random.uniform(-noise_mag, noise_mag, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Shrink radii to ensure feasibility after perturbation
        v_pert[2*N:] *= 0.93
        
        # Recompute feasible radii for the perturbed centers
        centers_pert = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = compute_feasible_radii(centers_pert) * 0.96
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
                current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract results
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        mr = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        cr[i] = min(cr[i], mr)
        
    # 2. Enforce non-overlap constraints iteratively
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
            
    return centers, cr, float(np.sum(cr))
