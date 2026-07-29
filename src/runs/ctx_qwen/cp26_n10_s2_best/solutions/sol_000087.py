# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000049 (state 0aad4082) state=59f6c7a5 sum of radii=2.614326 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute upper triangular indices for pairwise constraints
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Preallocate constraint array
    c = np.empty(N*4 + len(I_IDX))
    
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    d = np.hypot(dx, dy)
    c[4*N:] = d - (r[I_IDX] + r[J_IDX])
    
    return c

def jam_init(seed):
    """Generate a tightly packed initial configuration via physical relaxation."""
    np.random.seed(seed)
    c = np.random.uniform(0.1, 0.9, (N, 2))
    r = np.full(N, 0.015)
    
    for step in range(600):
        # Push overlapping circles apart
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-6:
                    overlap = r[i] + r[j] - d
                    ux, uy = dx / d, dy / d
                    c[i, 0] += 0.5 * overlap * ux
                    c[i, 1] += 0.5 * overlap * uy
                    c[j, 0] -= 0.5 * overlap * ux
                    c[j, 1] -= 0.5 * overlap * uy
        
        # Keep centers inside square
        c = np.clip(c, 0.001, 0.999)
        
        # Gradually expand radii towards max possible
        for i in range(N):
            max_r = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
            for j in range(N):
                if i == j: continue
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                if d * 0.5 < max_r:
                    max_r = d * 0.5
            # Exponential moving average towards max_r
            r[i] = 0.90 * r[i] + 0.10 * max_r
            
        # Small perturbation to avoid symmetric traps
        c += np.random.uniform(-0.0015, 0.0015, c.shape)
        c = np.clip(c, 0.001, 0.999)
        
    return c, r

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_val = -np.inf
    
    # Phase 1: Multi-start from jammed configurations
    for seed in range(12):
        c, r = jam_init(seed)
        v0 = np.concatenate([c[:, 0], c[:, 1], r])
        
        # Ensure initial feasibility for SLSQP
        cons_min = np.min(constraints(v0))
        if cons_min < -1e-3:
            scale = 0.98
            while np.min(constraints(np.concatenate([c[:, 0], c[:, 1], r * scale]))) < -1e-4:
                scale *= 0.98
            v0 = np.concatenate([c[:, 0], c[:, 1], r * scale])
            
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                if np.all(constraints(res.x) >= -1e-7):
                    best_val = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Aggressive perturbation & refinement of the best solution
    if best_v is not None:
        for iter_refine in range(10):
            v0 = best_v.copy()
            
            # Perturb centers more aggressively, shrink radii to create room
            perturbation_scale = 0.015 * (1.0 / (1.0 + 0.1 * iter_refine))
            v0[:2*N] += np.random.uniform(-perturbation_scale, perturbation_scale, 2*N)
            v0[:2*N] = np.clip(v0[:2*N], 0.01, 0.99)
            v0[2*N:] *= 0.94  # Shrink radii to guarantee feasibility after move
            
            try:
                res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_val:
                    if np.all(constraints(res.x) >= -1e-7):
                        best_val = -res.fun
                        best_v = res.x.copy()
            except Exception:
                continue

    # Fallback (should not be reached)
    if best_v is None:
        c, r = jam_init(0)
        best_v = np.concatenate([c[:, 0], c[:, 1], r * 0.9])
        
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    for i in range(N):
        margin = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], margin)
        radii[i] = max(radii[i], 0.0)
        
    # 2. Enforce non-overlap constraints iteratively
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d) * 0.5 + 1e-8
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
