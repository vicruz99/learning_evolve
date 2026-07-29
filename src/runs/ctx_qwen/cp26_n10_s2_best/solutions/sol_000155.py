# sol_000155 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000095 (state c7e336c8) state=b59b6667 sum of radii=2.630729 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for efficient vectorized constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

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
    
    # Boundary constraints: circles must be inside [0, 1]x[0, 1]
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c_pair = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c, c_pair])

def make_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:].copy()
    
    # Enforce boundary constraints
    r = np.minimum(r, x)
    r = np.minimum(r, 1.0 - x)
    r = np.minimum(r, y)
    r = np.minimum(r, 1.0 - y)
    
    # Enforce non-overlap constraints iteratively
    for _ in range(10):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        dist = np.sqrt(dx**2 + dy**2)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        if np.max(overlap) < 1e-10:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return np.concatenate([x, y, r])

def generate_configs():
    """Generates a diverse set of initial configurations."""
    configs = []
    
    # Hexagonal patterns with different row distributions
    row_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 6, 6, 4],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [7, 5, 5, 5, 4]
    ]
    
    for pat in row_patterns:
        if sum(pat) < N:
            continue
        r0 = 0.095
        pts = []
        y = r0
        row_idx = 0
        for count in pat:
            x_start = r0 if row_idx % 2 == 0 else 2.0 * r0
            for k in range(count):
                if len(pts) >= N:
                    break
                x = x_start + k * 2.0 * r0
                pts.append([x, y])
            y += r0 * np.sqrt(3.0)
            row_idx += 1
            
        pts = np.array(pts[:N])
        
        # Add variations: rotation, shift
        for angle in [0.0, 0.08, -0.08]:
            for sx in [-0.02, 0.0, 0.02]:
                for sy in [-0.02, 0.0, 0.02]:
                    p = pts.copy()
                    if angle != 0.0:
                        c, s = np.cos(angle), np.sin(angle)
                        p = (p - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
                    p += [sx, sy]
                    p = np.clip(p, 0.02, 0.98)
                    
                    # Varied initial radii to break symmetry
                    r = 0.04 + np.random.uniform(-0.002, 0.002, N)
                    v = np.concatenate([p[:,0], p[:,1], r])
                    configs.append(make_feasible(v))
                    
    # Random repelled points
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        # Simple repulsion relaxation
        for _ in range(80):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.2 and d > 1e-5:
                        f = (0.2 - d) * 0.5 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.01
            pts = np.clip(pts, 0.05, 0.95)
        r = 0.04 + np.random.uniform(-0.002, 0.002, N)
        v = np.concatenate([pts[:,0], pts[:,1], r])
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
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            
            s = -res.fun
            c = constraints(res.x)
            if np.min(c) >= -1e-6 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = configs[0]
        
    # Phase 2: Iterative shrink & perturb to escape local minima
    current_v = best_v.copy()
    for step in range(12):
        pert = current_v.copy()
        pert[2*N:] *= 0.975  # Shrink radii to create room for center movement
        pert[:2*N] += np.random.uniform(-0.003, 0.003, 2*N)
        pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
        pert = make_feasible(pert)
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            
            s = -res.fun
            c = constraints(res.x)
            if np.min(c) >= -1e-6 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
                current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract results
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    eps = 1e-8
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        cr[i] = min(cr[i], cx[i] - eps, 1.0 - cx[i] - eps, 
                    cy[i] - eps, 1.0 - cy[i] - eps)
        cr[i] = max(cr[i], 0.0)
        
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(20):
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
