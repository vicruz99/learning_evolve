# sol_000193 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000144 (state c0d23801) state=1f501298 sum of radii=2.630179 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dr = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - dr**2
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
    return np.clip(r * 0.80, 1e-6, 0.5)

def generate_hex_config(rows, scale=1.0, shift=0.0, angle=0.0):
    """Generates a hexagonal lattice configuration with specified row counts."""
    pts = []
    y = 0.1
    for r_idx, count in enumerate(rows):
        x_start = 0.1 + (r_idx % 2) * 0.1
        for c in range(count):
            pts.append([x_start + c * 0.2, y])
        y += np.sqrt(3) / 2 * 0.2
    pts = np.array(pts[:N])
    
    if scale != 1.0:
        pts = (pts - 0.5) * scale + 0.5
    if shift != 0.0:
        pts[:, 0] += shift
        
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        
    pts = np.clip(pts, 0.02, 0.98)
    return pts

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal row patterns (sums to 26)
    patterns = [
        (5,6,5,6,4), (6,5,6,5,4), (6,6,5,5,4), (5,5,6,5,5), (4,6,6,6,4),
        (5,6,4,6,5), (6,4,6,5,5), (5,7,5,5,4), (4,5,6,6,5)
    ]
    for pat in patterns:
        for scale in [0.95, 1.0, 1.05]:
            for angle in [-0.1, 0.0, 0.1]:
                configs.append(generate_hex_config(pat, scale=scale, angle=angle))
                
    # 2. Standard hex lattice with varying base radii
    for r0 in np.linspace(0.08, 0.11, 4):
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        configs.append(np.array(pts[:N]))
        
    # 3. Force-directed repelled starts
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(100):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 0.22 and d > 1e-4:
                        f = (0.22 - d) / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for centers in inits:
        r_init = get_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        centers_fallback = np.random.uniform(0.2, 0.8, (N, 2))
        radii_fallback = get_feasible_radii(centers_fallback)
        best_v = np.concatenate([centers_fallback[:, 0], centers_fallback[:, 1], radii_fallback])
        best_sum = -np.sum(radii_fallback)
        
    # Phase 2: Perturbation & Refinement to escape local minima
    current_v = best_v.copy()
    for step in range(30):
        np.random.seed(step + 100)
        v_pert = current_v.copy()
        
        # Shrink radii progressively to create space for center rearrangement
        shrink = 0.90 - step * 0.015
        v_pert[2*N:] *= max(0.70, shrink)
        
        # Perturb centers with decaying noise
        noise_scale = 0.006 * (1.0 - step / 30.0)
        v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Re-feasibilize radii based on perturbed centers
        centers_pert = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = get_feasible_radii(centers_pert) * 0.85
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = -res.fun
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
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
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
