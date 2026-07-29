# sol_000166 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000163 (state b9c973b0) state=5af08d8b sum of radii=2.628315 correctness=1.0
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
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    r_sum = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - r_sum**2
    
    return c

def make_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    cx = v[:N]
    cy = v[N:2*N]
    cr = v[2*N:].copy()
    
    # Enforce boundary constraints
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # Enforce non-overlap constraints iteratively
    for _ in range(15):
        dx = cx[PAIR_I] - cx[PAIR_J]
        dy = cy[PAIR_I] - cy[PAIR_J]
        dist = np.hypot(dx, dy)
        overlap = (cr[PAIR_I] + cr[PAIR_J]) - dist
        if np.max(overlap) < 1e-12:
            break
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-12
        cr[PAIR_I] = np.maximum(0.0, cr[PAIR_I] - shrink)
        cr[PAIR_J] = np.maximum(0.0, cr[PAIR_J] - shrink)
        
    return np.concatenate([cx, cy, cr])

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with various base radii and rotations
    for r0 in np.linspace(0.08, 0.11, 7):
        for angle in np.linspace(-0.3, 0.3, 9):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 5:
                x_start = r0 + (row % 2) * r0
                x = x_start
                while x <= 1.0 - r0 and len(pts) < N + 5:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3.0) * r0
                row += 1
            pts = np.array(pts[:N])
            
            # Rotate around center
            if angle != 0.0:
                c_val, s_val = np.cos(angle), np.sin(angle)
                pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
                
            # Add small jitter to break symmetry
            pts += np.random.uniform(-0.004, 0.004, pts.shape)
            pts = np.clip(pts, 0.02, 0.98)
            configs.append(pts)
            
    # 2. Perturbed square grids
    for s in np.linspace(0.12, 0.20, 6):
        pts = np.array([[i*s + s/2, j*s + s/2] for i in range(6) for j in range(5)])[:N]
        pts += np.random.uniform(-0.005, 0.005, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        configs.append(pts)
        
    # 3. Force-relaxed random configurations
    for seed in range(25):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        # Simple repulsion relaxation
        for _ in range(150):
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dist, np.inf)
            force = np.zeros_like(pts)
            mask = dist < 0.22
            f_mag = np.where(mask, (0.22 - dist) * 2.0 / dist, 0.0)
            force += np.sum(diff * f_mag[:, :, np.newaxis], axis=1)
            pts += force * 0.005
            pts = np.clip(pts, 0.02, 0.98)
        configs.append(pts)
        
    return configs

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.25)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    inits = generate_initial_configs()
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization
    for c_init in inits:
        # Start with moderate radii and ensure strict feasibility
        r_init = np.full(N, 0.04)
        v0 = make_feasible(np.concatenate([c_init[:,0], c_init[:,1], r_init]))
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 25000, 'ftol': 1e-14, 'disp': False})
            
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        return np.zeros((N, 2)), np.zeros(N), 0.0
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(40):
        np.random.seed(step + 2000)
        v_pert = current_v.copy()
        
        # Decaying noise for centers
        noise_scale = 0.006 * (1.0 - step / 40.0)
        v_pert[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        
        # Shrink radii to create breathing room for rearrangement
        v_pert[2*N:] *= 0.955
        v_pert = make_feasible(v_pert)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 18000, 'ftol': 1e-14, 'disp': False})
            
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
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    centers = np.column_stack((cx, cy))
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - 1e-11:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-11
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, cr, float(np.sum(cr))
