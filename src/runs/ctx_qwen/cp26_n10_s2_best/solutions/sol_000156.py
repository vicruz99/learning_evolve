# sol_000156 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=2200db75 sum of radii=2.631094 correctness=1.0
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
    """Compute inequality constraints: boundaries and pairwise non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    return np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        (x[PAIR_I] - x[PAIR_J])**2 + (y[PAIR_I] - y[PAIR_J])**2 - (r[PAIR_I] + r[PAIR_J])**2
    ])

def force_directed_init(seed):
    """Generate a well-spread initial configuration using repulsion forces."""
    np.random.seed(seed)
    centers = np.random.uniform(0.1, 0.9, (N, 2))
    target = 0.2
    lr = 0.05
    for _ in range(600):
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                d = np.hypot(diff[0], diff[1])
                if d < target and d > 1e-5:
                    f = (target - d) / d * lr
                    forces[i] += f * diff
                    forces[j] -= f * diff
        centers += forces
        centers = np.clip(centers, 0.02, 0.98)
        lr *= 0.995
    return centers

def ensure_strict_feasibility(v):
    """Adjust radii to guarantee constraints are strictly satisfied."""
    cx = v[:N].copy()
    cy = v[N:2*N].copy()
    cr = v[2*N:].copy()
    
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if d < cr[i] + cr[j]:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-7
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
    return np.concatenate([cx, cy, cr])

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = []
    
    # 1. Force-directed layouts for uniform spread
    for seed in range(15):
        c = force_directed_init(seed)
        inits.append(np.concatenate([c[:,0], c[:,1], np.full(N, 0.025)]))
        
    # 2. Hexagonal lattices with varying densities and jitter
    for r0 in [0.08, 0.09, 0.10, 0.105]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N:
            x_start = r0 if row % 2 == 0 else 2*r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N:
                pts.append([x, y])
                x += 2*r0
            y += np.sqrt(3)*r0
            row += 1
        c = np.array(pts[:N])
        c += np.random.uniform(-0.01, 0.01, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(np.concatenate([c[:,0], c[:,1], np.full(N, 0.025)]))
        
    # 3. Staggered grids
    for s in [0.0, 0.015, 0.03]:
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08+s + i*0.16, 0.08+s + j*0.20])
        c = np.array(pts[:N])
        c += np.random.uniform(-0.01, 0.01, c.shape)
        c = np.clip(c, 0.02, 0.98)
        inits.append(np.concatenate([c[:,0], c[:,1], np.full(N, 0.025)]))

    # Primary optimization pass
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                if np.all(constraints(res.x) >= -1e-6):
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass

    # Refinement loop: shrink, perturb, re-optimize to escape local minima
    if best_v is not None:
        curr = best_v.copy()
        for step in range(20):
            np.random.seed(step + 100)
            pert = curr.copy()
            pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            pert[2*N:] *= 0.90 # Shrink to create space for rearrangement
            
            # Quick feasibility fix before optimization
            pert = ensure_strict_feasibility(pert)
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    if np.all(constraints(res.x) >= -1e-6):
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        curr = best_v.copy()
            except Exception:
                pass

    # Extract and strictly enforce validation constraints
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Boundary enforcement
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # Non-overlap enforcement
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
