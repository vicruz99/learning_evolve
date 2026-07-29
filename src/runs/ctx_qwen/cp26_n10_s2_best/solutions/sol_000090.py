# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000049 (state 0aad4082) state=6356247a sum of radii=2.624052 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and non-overlap. Returns array where all elements >= 0."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c = [
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ]
    
    # Pairwise non-overlap constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c.append(dist[mask] - r_sum[mask])
    
    return np.concatenate(c)

def make_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    # Enforce boundary constraints
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        
    # Enforce non-overlap constraints iteratively
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if r[i] + r[j] > d:
                    shrink = (r[i] + r[j] - d) / 2.0 + 1e-9
                    r[i] = max(0.0, r[i] - shrink)
                    r[j] = max(0.0, r[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return np.concatenate([x, y, r])

def force_directed_init(seed):
    """Generates a dense, valid initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    cx = np.random.uniform(0.15, 0.85, N)
    cy = np.random.uniform(0.15, 0.85, N)
    r = np.full(N, 0.05)
    
    for _ in range(800):
        r += 0.00008
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Compute repulsion forces only for close pairs
        mask = d < r + r[None, :]
        f = (r + r[None, :] - d) / (d**2 + 1e-8)
        f *= mask
        
        fcx = np.sum(dx * f, axis=1)
        fcy = np.sum(dy * f, axis=1)
        
        cx += fcx * 0.002
        cy += fcy * 0.002
        cx = np.clip(cx, 0.02, 0.98)
        cy = np.clip(cy, 0.02, 0.98)
        
    return make_feasible(np.concatenate([cx, cy, r]))

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Generate diverse initial configurations
    starts = []
    for s in range(15):
        starts.append(force_directed_init(s))
        
    # Add perturbed hexagonal and grid starts
    for s in range(10):
        np.random.seed(s)
        c_hex = []
        y = 0.05
        row = 0
        while len(c_hex) < N:
            x = 0.05 if row % 2 == 0 else 0.11
            while x <= 0.95 and len(c_hex) < N:
                c_hex.append([x + np.random.uniform(-0.02, 0.02), 
                              y + np.random.uniform(-0.02, 0.02)])
                x += 0.17
            y += 0.14
            row += 1
        v_hex = np.concatenate([np.array(c_hex[:N])[:, 0], np.array(c_hex[:N])[:, 1], np.full(N, 0.03)])
        starts.append(make_feasible(v_hex))
        
    # Primary optimization pass
    for v0 in starts:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.all(constraints(res.x) >= -1e-6):
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Local search refinement to escape shallow local minima
    if best_v is not None:
        for _ in range(12):
            vp = best_v.copy()
            # Perturb centers
            vp[:2 * N] += np.random.uniform(-0.004, 0.004, 2 * N)
            vp[:2 * N] = np.clip(vp[:2 * N], 0.01, 0.99)
            # Slightly scale radii up to push out of tight constraints
            vp[2 * N:] *= 1.002
            vp = make_feasible(vp)
            
            try:
                res = minimize(objective, vp, method='SLSQP', bounds=bounds, constraints=cons_dict,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_sum and np.all(constraints(res.x) >= -1e-6):
                    best_sum = -res.fun
                    best_v = res.x.copy()
            except Exception:
                continue
                
    if best_v is None:
        # Fallback safe configuration
        best_v = make_feasible(np.concatenate([np.random.uniform(0.1, 0.9, N), 
                                               np.random.uniform(0.1, 0.9, N), 
                                               np.full(N, 0.02)]))
        
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator compliance
    for i in range(N):
        cr[i] = min(cr[i], cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if cr[i] + cr[j] > d:
                shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-7
                cr[i] = max(0.0, cr[i] - shrink)
                cr[j] = max(0.0, cr[j] - shrink)
                
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
