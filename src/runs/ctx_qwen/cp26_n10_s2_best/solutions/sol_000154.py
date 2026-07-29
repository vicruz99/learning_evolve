# sol_000154 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000095 (state c7e336c8) state=9ffe3e68 sum of radii=2.625913 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap (squared for smooth gradients)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def get_feasible_radii(centers):
    """Compute strictly feasible initial radii based on current centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        r[i] = min(centers[i, 0], 1.0 - centers[i, 0], 
                   centers[i, 1], 1.0 - centers[i, 1])
        for j in range(N):
            if i != j:
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if d / 2.0 < r[i]:
                    r[i] = d / 2.0
    return r * 0.78  # Scale down to guarantee strict feasibility

def generate_initial_configs():
    """Generate a wide variety of initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with random rotations and shifts
    for _ in range(35):
        r0 = np.random.uniform(0.065, 0.105)
        angle = np.random.uniform(-0.35, 0.35)
        shift = np.random.uniform(-0.06, 0.06, 2)
        
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 12:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
            
        pts = np.array(pts)
        pts -= 0.5
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, -s], [s, c]]) + 0.5 + shift
        
        mask = (pts[:, 0] > 0.04) & (pts[:, 0] < 0.96) & \
               (pts[:, 1] > 0.04) & (pts[:, 1] < 0.96)
        valid = pts[mask]
        if len(valid) >= N:
            idx = np.random.choice(len(valid), N, replace=False)
            centers = valid[idx] + np.random.uniform(-0.004, 0.004, (N, 2))
            configs.append(centers)
            
    # 2. Random dense scatters
    for _ in range(25):
        centers = np.random.uniform(0.06, 0.94, (N, 2))
        configs.append(centers)
        
    # 3. Structured grids with jitter
    for _ in range(15):
        gx = np.linspace(0.08, 0.92, 6)
        gy = np.linspace(0.08, 0.92, 5)
        cx, cy = np.meshgrid(gx, gy)
        pts = np.column_stack([cx.flatten(), cy.flatten()])
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx] + np.random.uniform(-0.025, 0.025, (N, 2))
        centers = np.clip(centers, 0.05, 0.95)
        configs.append(centers)
        
    # 4. Corner-focused clusters
    for _ in range(10):
        corners = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]])
        rest = np.random.uniform(0.25, 0.75, (N - 4, 2))
        centers = np.vstack([corners, rest])
        centers += np.random.uniform(-0.03, 0.03, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        configs.append(centers)
        
    return configs

def run_packing():
    """Optimizes packing of 26 circles in a unit square to maximize sum of radii."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start exploration
    configs = generate_initial_configs()
    for centers in configs:
        r_init = get_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            s = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-8 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        v0 = np.concatenate([np.random.uniform(0.1, 0.9, 2*N), np.full(N, 0.05)])
        best_v = v0
        best_sum = -np.sum(v0[2*N:])
        
    # Phase 2: Perturbation & Refinement to escape local minima
    current_v = best_v.copy()
    for step in range(20):
        pert = current_v.copy()
        # Shrink radii to create room for center rearrangement
        shrink_factor = 0.88 - step * 0.008
        pert[2*N:] *= max(0.65, shrink_factor)
        
        # Perturb centers
        noise_scale = 0.006 - step * 0.0002
        pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        pert[:2*N] = np.clip(pert[:2*N], 0.02, 0.98)
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
            s = -res.fun
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-8 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
                current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract results
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        max_r = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if cr[i] > max_r:
            cr[i] = max_r
            
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
