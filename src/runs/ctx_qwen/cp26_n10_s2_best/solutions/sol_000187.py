# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000155 (state b59b6667) state=a88ec6ce sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Compute inequality constraints: boundaries and pairwise non-overlap.
    Uses squared distances for smoother gradients and numerical stability.
    All elements must be >= 0.
    """
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
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:].copy()
    
    # Enforce boundary constraints
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    
    # Enforce non-overlap constraints iteratively
    for _ in range(8):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        if np.max(overlap) < 1e-11:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-10
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return np.concatenate([x, y, r])

def force_directed_init(seed, iters=400):
    """Generate a well-spread initial configuration using repulsive forces."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.15, 0.85, (N, 2))
    for step in range(iters):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        mask = dists < 0.22
        f_mag = np.zeros_like(dists)
        f_mag[mask] = (0.22 - dists[mask]) / dists[mask]
        
        forces = np.sum(diff * f_mag[:, :, np.newaxis], axis=1)
        pts += forces * 0.015 * (1.0 - step / iters)
        pts = np.clip(pts, 0.04, 0.96)
    return pts

def generate_configs():
    """Generates a diverse set of initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying parameters
    for r0 in np.linspace(0.088, 0.112, 6):
        for angle in np.linspace(-0.12, 0.12, 5):
            for sx, sy in [(-0.015, -0.015), (0.0, 0.0), (0.015, 0.015)]:
                pts = []
                y = r0 + sy
                row = 0
                while len(pts) < N + 6:
                    x_start = r0 + sx + (row % 2) * r0
                    x = x_start
                    while x <= 1.0 - r0 and len(pts) < N + 6:
                        pts.append([x, y])
                        x += 2.0 * r0
                    y += r0 * np.sqrt(3.0)
                    row += 1
                pts = np.array(pts[:N])
                
                c, s = np.cos(angle), np.sin(angle)
                pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
                pts = np.clip(pts, 0.02, 0.98)
                
                r_init = np.full(N, 0.035)
                v = np.concatenate([pts[:, 0], pts[:, 1], r_init])
                configs.append(make_feasible(v))

    # 2. Force-directed layouts
    for seed in range(12):
        pts = force_directed_init(seed)
        r_init = np.full(N, 0.035)
        v = np.concatenate([pts[:, 0], pts[:, 1], r_init])
        configs.append(make_feasible(v))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    configs = generate_configs()
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start exploration
    for v0 in configs:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            s = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = make_feasible(configs[0])
        
    # Phase 2: Iterative refinement to escape local minima
    current_v = best_v.copy()
    for step in range(35):
        np.random.seed(step + 900)
        v_p = current_v.copy()
        
        # Decaying noise for center perturbation
        noise_scale = 0.004 * max(0.0, 1.0 - step / 35.0)
        v_p[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Occasionally swap two centers to break symmetry traps
        if step % 5 == 0:
            idx = np.random.choice(N, 2, replace=False)
            v_p[idx[0]], v_p[idx[1]] = v_p[idx[1]], v_p[idx[0]]
            v_p[N+idx[0]], v_p[N+idx[1]] = v_p[N+idx[1]], v_p[N+idx[0]]
            
        # Shrink radii to create slack for rearrangement
        shrink_factor = 0.91 - step * 0.0015
        v_p[2*N:] *= max(0.7, shrink_factor)
        v_p = make_feasible(v_p)
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            s = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
                current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    eps = 1e-9
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - eps:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + eps
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
