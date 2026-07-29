# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000055 (state f6ce444f) state=c41af558 sum of radii=2.609603 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dist - r_sum
    
    return np.concatenate([c_bound, c_pair])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Stage 1: Multi-start exploration from perturbed hexagonal lattices
    for seed in range(30):
        np.random.seed(seed)
        r0 = 0.095
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 10:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1 - r0:
                pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        pts = np.array(pts)
        idx = np.random.choice(len(pts), N, replace=False)
        centers = pts[idx].copy()
        
        # Add jitter to break symmetry and explore basins
        centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        # Initial radii small enough to guarantee feasibility
        r_init = np.full(N, 0.04)
        x0 = np.concatenate([centers[:,0], centers[:,1], r_init])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                # Verify feasibility
                if np.all(constraints(res.x) >= -1e-6):
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Stage 2: Basin hopping / perturbation refinement to escape local minima
    if best_v is not None:
        for _ in range(40):
            v_pert = best_v.copy()
            # Perturb centers
            v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
            # Shrink radii to ensure feasibility after perturbation
            v_pert[2*N:] *= 0.95
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    if np.all(constraints(res.x) >= -1e-6):
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Extract results
    if best_v is None:
        centers = np.random.uniform(0.2, 0.8, (N, 2))
        radii = np.full(N, 0.04)
    else:
        centers = np.column_stack((best_v[:N], best_v[N:2*N]))
        radii = best_v[2*N:]
        
    # Strict post-processing to guarantee validation passes
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
