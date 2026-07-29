# sol_000143 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=f3956e2b sum of radii=2.630179 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def make_initial_config(seed, layout='hex'):
    """Generate a feasible initial configuration."""
    np.random.seed(seed)
    if layout == 'hex':
        r0 = 0.085 + seed * 0.002
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 10:
            x = r0 if row % 2 == 0 else 2.0 * r0
            while x <= 1.0 - r0:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts[:N])
        # Random rotation to break symmetry and explore boundary fitting
        angle = np.random.uniform(-12, 12) * np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
        pts = np.clip(pts, 0.03, 0.97)
    else:
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        # Quick repulsion to spread points
        for _ in range(150):
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
                    if d < 0.25 and d > 1e-5:
                        push = (0.25 - d) / d
                        diff = pts[i] - pts[j]
                        pts[i] += push * diff * 0.15
                        pts[j] -= push * diff * 0.15
            pts = np.clip(pts, 0.03, 0.97)
            
    r_init = np.full(N, 0.045)
    return np.concatenate([pts[:, 0], pts[:, 1], r_init])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # 1. Generate diverse initial configurations
    inits = []
    for seed in range(25):
        inits.append(make_initial_config(seed, 'hex'))
    for seed in range(10):
        inits.append(make_initial_config(seed, 'rand'))
        
    # Phase 1: Multi-start optimization
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if -res.fun > best_sum:
                # Accept if sufficiently feasible
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Adaptive perturbation refinement to escape local minima
    if best_v is not None:
        current_v = best_v
        for step in range(40):
            np.random.seed(step + 1000)
            pert = current_v.copy()
            # Gradually shrink radii to create space for center rearrangement
            shrink_factor = 0.995 - step * 0.0003
            pert[2*N:] *= max(0.97, shrink_factor)
            
            # Perturb centers
            noise_scale = 0.004 * (1.0 - step * 0.02)
            noise = np.random.uniform(-noise_scale, noise_scale, 2*N)
            pert[:2*N] += noise
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v
            except Exception:
                pass
                
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage and safety margin
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
