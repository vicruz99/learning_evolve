# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000053 (state 2e035c71) state=90c9a519 sum of radii=2.606590 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute upper triangular mask for pairwise constraints
TRI_MASK = np.triu(np.ones((N, N), dtype=bool), k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and pairwise non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    c.append((dx**2 + dy**2)[TRI_MASK] - dr[TRI_MASK]**2)
    
    return np.concatenate(c)

def make_feasible(cx, cy, r, shrink=0.9):
    """Shrink radii until the configuration strictly satisfies all constraints."""
    for i in range(N):
        mr = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if r[i] > mr:
            r[i] = mr
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            mr = d / 2.0
            if r[i] > mr: r[i] = mr
            if r[j] > mr: r[j] = mr
    r *= shrink
    return r

def get_hex_init(seed):
    """Generate a structured hexagonal lattice initialization."""
    np.random.seed(seed)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [7,5,6,5,3]]
    pat = patterns[seed % len(patterns)]
    r0 = 0.085
    cx, cy = [], []
    y = r0
    for idx, cnt in enumerate(pat):
        x_start = r0 + (idx % 2) * r0
        for _ in range(cnt):
            cx.append(x_start)
            cy.append(y)
            x_start += 2 * r0
        y += np.sqrt(3) * r0
        
    cx = np.array(cx[:N])
    cy = np.array(cy[:N])
    # Normalize to fit comfortably inside the square
    cx = (cx - cx.min()) / (cx.max() - cx.min()) * 0.8 + 0.1
    cy = (cy - cy.min()) / (cy.max() - cy.min()) * 0.8 + 0.1
    r = np.full(N, 0.05)
    r = make_feasible(cx, cy, r, 0.8)
    return cx, cy, r

def get_sim_init(seed):
    """Generate an initialization via growing-circle simulation."""
    np.random.seed(seed)
    cx = np.random.uniform(0.15, 0.85, N)
    cy = np.random.uniform(0.15, 0.85, N)
    r = np.full(N, 0.02)
    
    # Iteratively grow circles and resolve collisions
    for _ in range(500):
        r += 0.00025
        for i in range(N):
            # Push inward if hitting boundaries
            cx[i] = np.clip(cx[i], r[i], 1.0 - r[i])
            cy[i] = np.clip(cy[i], r[i], 1.0 - r[i])
            
            for j in range(i + 1, N):
                dx = cx[j] - cx[i]
                dy = cy[j] - cy[i]
                d = np.hypot(dx, dy)
                if d < r[i] + r[j] and d > 1e-9:
                    overlap = r[i] + r[j] - d
                    nx, ny = dx / d, dy / d
                    # Push apart slightly more than overlap to ensure separation
                    cx[i] -= nx * overlap * 0.51
                    cy[i] -= ny * overlap * 0.51
                    cx[j] += nx * overlap * 0.51
                    cy[j] += ny * overlap * 0.51
    r = make_feasible(cx, cy, r, 0.85)
    return cx, cy, r

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_val = -1.0
    
    # Stage 1: Multi-start optimization from diverse configurations
    for i in range(12):
        cx, cy, r = get_hex_init(i)
        v0 = np.concatenate([cx, cy, r])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val and np.min(constraints(res.x)) >= -1e-6:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    for i in range(10):
        cx, cy, r = get_sim_init(i)
        v0 = np.concatenate([cx, cy, r])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val and np.min(constraints(res.x)) >= -1e-6:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback if all optimizations fail
    if best_v is None:
        cx, cy, r = get_hex_init(0)
        best_v = np.concatenate([cx, cy, r])
        
    # Stage 2: Perturbation search to escape local minima
    for step in range(8):
        np.random.seed(step * 99 + 7)
        v0 = best_v.copy()
        # Perturb centers randomly
        v0[:2*N] += np.random.uniform(-0.004, 0.004, 2 * N)
        v0[:2*N] = np.clip(v0[:2*N], 0.02, 0.98)
        # Shrink radii to restore feasibility after perturbation
        v0[2*N:] *= 0.96
        v0[2*N:] = make_feasible(v0[:N], v0[N:2*N], v0[2*N:], 0.9)
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val and np.min(constraints(res.x)) >= -1e-6:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass

    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Stage 3: Strict post-processing to guarantee validator compliance
    # Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    
    # Enforce non-overlap constraints iteratively with safety margin
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < cr[i] + cr[j] - 1e-9:
                    shr = (cr[i] + cr[j] - d) / 2.0 + 1e-8
                    cr[i] = max(0.0, cr[i] - shr)
                    cr[j] = max(0.0, cr[j] - shr)
                    changed = True
        if not changed:
            break
            
    return np.column_stack((cx, cy)), cr, float(np.sum(cr))
