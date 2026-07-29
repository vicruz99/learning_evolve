# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000144 (state c0d23801) state=443cbb20 sum of radii=2.626678 correctness=1.0
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
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    x = centers[:, 0]
    y = centers[:, 1]
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    r = np.minimum(wall, min_dists / 2.0)
    return np.clip(r * 0.94, 1e-4, 0.5)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    configs = []
    
    # 1. Structured row distributions (mimics optimal hex packings for N=26)
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4], [4, 6, 6, 5, 5], [6, 4, 6, 5, 5]
    ]
    for pat in patterns:
        pts = []
        y = 0.08
        row_h = 0.165
        for r_idx, count in enumerate(pat):
            spacing = 1.0 / (count + 1)
            for c in range(count):
                pts.append([spacing * (c + 1), y + row_h * r_idx])
        pts = np.array(pts)
        pts += np.random.uniform(-0.008, 0.008, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
    # 2. Rotated Hexagonal Lattices
    for angle in np.linspace(-0.35, 0.35, 11):
        r0 = 0.092 + np.random.uniform(-0.005, 0.005)
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 8:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 8:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        c_val, s_val = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
        pts = np.clip(pts, 0.02, 0.98)
        configs.append(pts)
        
    # 3. Force-relaxed dense starts
    for seed in range(12):
        np.random.seed(seed)
        pts = np.random.uniform(0.12, 0.88, (N, 2))
        for _ in range(250):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.22 and d > 1e-4:
                        f = (0.22 - d) * 1.2 / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.04, 0.96)
        configs.append(pts)
        
    # Phase 1: Multi-start optimization
    for centers in configs:
        r_init = get_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        # Fallback initialization
        centers_fallback = np.random.uniform(0.15, 0.85, (N, 2))
        radii_fallback = get_feasible_radii(centers_fallback)
        best_v = np.concatenate([centers_fallback[:, 0], centers_fallback[:, 1], radii_fallback])
        best_sum = -np.sum(radii_fallback)
        
    # Phase 2: Adaptive refinement to escape local minima
    current_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 500)
        v_pert = current_v.copy()
        
        # Decaying noise schedule
        noise_scale = 0.004 * (1.0 - 0.015 * step)
        v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.015, 0.985)
        
        # Shrink radii to create space for center rearrangement
        centers_pert = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = get_feasible_radii(centers_pert) * 0.91
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing for validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
