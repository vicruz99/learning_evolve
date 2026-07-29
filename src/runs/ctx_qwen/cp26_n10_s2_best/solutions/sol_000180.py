# sol_000180 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000167 (state d81766f0) state=cf8312d8 sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.hypot(dx, dy)
    c[4*N:] = dist - r[PAIR_I] - r[PAIR_J]
    
    return c

def compute_initial_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
    dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
    dists = np.hypot(dx, dy)
    r_pairs = dists / 2.0
    
    for k in range(NUM_PAIRS):
        i, j = PAIR_I[k], PAIR_J[k]
        val = r_pairs[k]
        if val < r[i]: r[i] = val
        if val < r[j]: r[j] = val
        
    return np.clip(r * 0.85, 0.001, 0.25)

def generate_hexagonal_configs():
    """Generate hexagonal lattice configurations with rotations and shifts."""
    configs = []
    for seed in range(20):
        np.random.seed(seed)
        r0 = 0.085 + np.random.uniform(-0.005, 0.005)
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        sx = np.random.uniform(-0.02, 0.02)
        sy = np.random.uniform(-0.02, 0.02)
        
        pts = []
        y = r0 + sy
        row = 0
        while len(pts) < N + 10:
            x_start = r0 + sx + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 10:
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3.0)
            row += 1
            
        pts = np.array(pts[:N])
        c_val, s_val = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
        pts = np.clip(pts, 0.03, 0.97)
        configs.append(pts)
    return configs

def generate_force_relaxed_configs():
    """Generate force-relaxed random configurations."""
    configs = []
    for seed in range(15):
        np.random.seed(seed + 1000)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(200):
            forces = np.zeros_like(pts)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            mask = dists < 0.22
            f_mag = np.zeros_like(dists)
            f_mag[mask] = (0.22 - dists[mask]) / (dists[mask] + 1e-8)
            f_vec = diff * f_mag[:, :, np.newaxis]
            forces = np.sum(f_vec, axis=1)
            
            pts += forces * 0.04
            pts = np.clip(pts, 0.04, 0.96)
        configs.append(pts)
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.28)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = generate_hexagonal_configs() + generate_force_relaxed_configs()
    
    # Phase 1: Multi-start optimization with layout pre-conditioning
    for centers in inits:
        r_init = compute_initial_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            # Phase 1A: Optimize centers with fixed radii to find valid dense layout
            res_centers = minimize(lambda xc: np.sum((xc[2*N:] - v0[2*N:])**2), 
                                   v0, method='SLSQP', bounds=bounds, constraints=cons,
                                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            # Phase 1B: Joint optimization from the valid layout
            res = minimize(objective, res_centers.x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue

    if best_v is None:
        c_fb = np.random.uniform(0.15, 0.85, (N, 2))
        r_fb = compute_initial_radii(c_fb)
        best_v = np.concatenate([c_fb[:, 0], c_fb[:, 1], r_fb])
        best_sum = -np.sum(r_fb)

    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 5000)
        v_pert = current_v.copy()
        
        # Decaying noise schedule
        noise_scale = 0.004 * (1.0 - step / 40.0)
        v_pert[:2*N] += np.random.normal(0, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
        
        # Shrink radii to create slack and allow rearrangement
        shrink_factor = 0.90 - step * 0.001
        v_pert[2*N:] *= max(0.75, shrink_factor)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False})
            
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue

    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    # Validator allows dist >= r1 + r2 - 1e-12. We use 1e-13 buffer.
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d + 1e-13:
                    shrink = (radii[i] + radii[j] - d - 1e-13) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
