# sol_000152 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000095 (state c7e336c8) state=e8b37863 sum of radii=2.628596 correctness=1.0
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
    """Compute inequality constraints: boundaries and pairwise non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: circles inside [0, 1]x[0, 1]
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints (squared for stability)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def get_feasible_r(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.full(N, 0.5)
    # Distance to boundaries
    for i in range(N):
        r[i] = min(centers[i, 0], 1.0 - centers[i, 0], 
                   centers[i, 1], 1.0 - centers[i, 1])
    # Distance to other centers
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            val = d / 2.0
            if val < r[i]: r[i] = val
            if val < r[j]: r[j] = val
    return r * 0.80  # Leave slack for optimizer to expand

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    np.random.seed(42)
    
    # 1. Hexagonal lattices with varying densities
    for r0 in np.linspace(0.085, 0.110, 8):
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x_start = r0 if row % 2 == 0 else 2.0 * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.008, 0.008, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        r_init = get_feasible_r(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # 2. Force-directed repulsion layouts
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        # Repulsion relaxation to spread circles evenly
        for _ in range(150):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                    if d < 0.25 and d > 1e-4:
                        f = (0.25 - d) * 0.6 / d
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.05
            pts = np.clip(pts, 0.05, 0.95)
        r_init = get_feasible_r(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    # 3. Corner-focused placements
    for seed in range(10):
        np.random.seed(200 + seed)
        corners = np.array([[0.13, 0.13], [0.87, 0.13], [0.13, 0.87], [0.87, 0.87]])
        rest = np.random.uniform(0.20, 0.80, (N - 4, 2))
        pts = np.vstack([corners, rest])
        pts += np.random.uniform(-0.015, 0.015, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        r_init = get_feasible_r(pts)
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_init]))
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    configs = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for v0 in configs:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            
            curr_sum = -res.fun
            cons_vals = constraints(res.x)
            if np.min(cons_vals) >= -1e-6 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback initialization
    if best_v is None:
        best_v = configs[0]
        
    # Phase 2: Perturbation & Refinement Loop to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(18):
            np.random.seed(step + 500)
            v_pert = current_v.copy()
            
            # Gradually shrink radii to allow significant rearrangement
            shrink_factor = 0.88 - step * 0.005
            v_pert[2*N:] *= max(0.75, shrink_factor)
            
            # Perturb centers
            noise_scale = 0.008 - step * 0.0003
            v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                
                curr_sum = -res.fun
                cons_vals = constraints(res.x)
                if np.min(cons_vals) >= -1e-6 and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
            except Exception:
                continue
                
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        cr[i] = min(cr[i], cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        
    # 2. Enforce non-overlap constraints iteratively with safety margin
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
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
