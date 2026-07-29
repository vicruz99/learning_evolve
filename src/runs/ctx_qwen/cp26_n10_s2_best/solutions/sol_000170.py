# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000125 (state 42bc631b) state=69723765 sum of radii=2.628318 correctness=1.0
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
    """Compute inequality constraints: boundaries and squared non-overlap distances."""
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
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def make_feasible(v):
    """Iteratively adjust radii to guarantee strict feasibility."""
    cx = v[:N].copy()
    cy = v[N:2*N].copy()
    cr = v[2*N:].copy()
    
    # Enforce boundaries
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # Enforce pairwise non-overlap
    for _ in range(15):
        dx = cx[PAIR_I] - cx[PAIR_J]
        dy = cy[PAIR_I] - cy[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (cr[PAIR_I] + cr[PAIR_J]) - dist
        if np.max(overlap) < 1e-9:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        cr[PAIR_I] = np.maximum(0.0, cr[PAIR_I] - shrink)
        cr[PAIR_J] = np.maximum(0.0, cr[PAIR_J] - shrink)
        
    return np.concatenate([cx, cy, cr])

def generate_configs():
    """Generate a diverse set of strictly feasible initial configurations."""
    configs = []
    
    # 1. Rotated Hexagonal Lattices
    for seed in range(25):
        np.random.seed(seed)
        r0 = 0.090 + np.random.uniform(-0.015, 0.015)
        angle = np.random.uniform(-0.3, 0.3)
        sx = np.random.uniform(-0.05, 0.05)
        sy = np.random.uniform(-0.05, 0.05)
        
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
        pts = np.clip(pts, 0.02, 0.98)
        
        r_init = np.full(N, 0.04)
        v = np.concatenate([pts[:, 0], pts[:, 1], r_init])
        configs.append(make_feasible(v))
        
    # 2. Perturbed Grids
    for seed in range(15):
        np.random.seed(seed + 1000)
        pts = np.array([[0.10 + i*0.15 + np.random.uniform(-0.02, 0.02), 
                         0.10 + j*0.18 + np.random.uniform(-0.02, 0.02)] 
                        for i in range(6) for j in range(5)])[:N]
        pts = np.clip(pts, 0.02, 0.98)
        r_init = np.full(N, 0.04)
        v = np.concatenate([pts[:, 0], pts[:, 1], r_init])
        configs.append(make_feasible(v))
        
    # 3. Force-Relaxed Random Starts
    for seed in range(15):
        np.random.seed(seed + 2000)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        
        # Vectorized repulsion relaxation
        for _ in range(400):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            mask = (dist < 0.28) & (np.eye(N, dtype=bool) == False)
            
            f_mag = np.zeros_like(dist)
            f_mag[mask] = (0.28 - dist[mask]) * 0.5
            f_vec = diff * f_mag[:, :, np.newaxis] / (dist[:, :, np.newaxis] + 1e-12)
            forces = np.sum(f_vec, axis=1)
            
            pts += forces * 0.03
            pts = np.clip(pts, 0.05, 0.95)
            
        r_init = np.full(N, 0.035)
        v = np.concatenate([pts[:, 0], pts[:, 1], r_init])
        configs.append(make_feasible(v))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    configs = generate_configs()
    
    # Phase 1: Multi-start exploration
    for v0 in configs:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 18000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization fails completely
    if best_v is None:
        best_v = make_feasible(configs[0])
        best_sum = -np.sum(best_v[2*N:])
        
    # Phase 2: Basin-hopping refinement to escape shallow local minima
    current_v = best_v.copy()
    for step in range(45):
        np.random.seed(step + 5000)
        # Decaying noise schedule
        noise_scale = 0.0045 * np.exp(-step / 12.0)
        
        v_pert = current_v.copy()
        v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Slight radius inflation encourages expansion into newly opened gaps
        v_pert[2*N:] *= 1.003
        v_pert = make_feasible(v_pert)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
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
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    for _ in range(25):
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
