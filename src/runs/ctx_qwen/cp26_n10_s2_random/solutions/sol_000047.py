# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000044 (state 69bc282d) state=71058a9b sum of radii=2.621920 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def make_bounds():
    """Creates variable bounds: x,y in [0,1], r in [1e-7, 0.5]"""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def compute_constraints(p):
    """
    Computes all inequality constraint values (must be >= 0).
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Preallocate constraint array
    # 4 boundary constraints per circle + N*(N-1)/2 pairwise constraints
    n_con = 4 * N + N * (N - 1) // 2
    con = np.empty(n_con)
    idx = 0
    
    # Boundary constraints
    con[idx:idx+N] = x - r; idx += N
    con[idx:idx+N] = 1.0 - x - r; idx += N
    con[idx:idx+N] = y - r; idx += N
    con[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            con[idx] = dx*dx + dy*dy - dr*dr
            idx += 1
            
    return con

def objective(p):
    """Objective: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def repair(p):
    """
    Iteratively shrinks radii to resolve overlaps and clamps to boundaries.
    Shrinks radii proportionally to their size to preserve sum of radii.
    """
    x = p[0::3].copy()
    y = p[1::3].copy()
    r = p[2::3].copy()
    
    # Strict boundary enforcement
    r = np.minimum(r, x)
    r = np.minimum(r, 1.0 - x)
    r = np.minimum(r, y)
    r = np.minimum(r, 1.0 - y)
    
    # Overlap resolution
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.hypot(x[i] - x[j], y[i] - y[j])
                req = r[i] + r[j]
                if dist < req - 1e-12:
                    overlap = req - dist
                    total = r[i] + r[j]
                    if total > 1e-12:
                        fi = r[i] / total
                        fj = r[j] / total
                    else:
                        fi = 0.5
                        fj = 0.5
                    r[i] -= overlap * fi
                    r[j] -= overlap * fj
                    changed = True
        if not changed:
            break
            
    return x, y, np.maximum(r, 1e-7)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Main function to pack 26 circles in a unit square."""
    np.random.seed(42)
    bounds = make_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_p = None
    best_sum = -1.0
    
    inits = []
    
    # 1. Hexagonal lattice initialization (high density baseline)
    r0 = 0.095
    cx, cy = [], []
    y = r0
    row = 0
    while len(cx) < N:
        sx = r0 if row % 2 == 0 else 2 * r0
        x = sx
        while x + r0 <= 1.0 + 1e-9 and len(cx) < N:
            cx.append(x)
            cy.append(y)
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
    cx, cy = np.array(cx), np.array(cy)
    
    # Add noise variants to break symmetry
    for noise in [0.0, 0.001, 0.003, 0.005, 0.01]:
        nc = cx + np.random.normal(0, noise, N)
        ny = cy + np.random.normal(0, noise, N)
        nc = np.clip(nc, 0.02, 0.98)
        ny = np.clip(ny, 0.02, 0.98)
        p0 = np.zeros(3 * N)
        p0[0::3] = nc
        p0[1::3] = ny
        p0[2::3] = r0
        inits.append(p0)
        
    # 2. Random dense configurations
    for seed in range(15):
        rng = np.random.default_rng(seed)
        p0 = np.zeros(3 * N)
        p0[0::3] = rng.uniform(0.1, 0.9, N)
        p0[1::3] = rng.uniform(0.1, 0.9, N)
        p0[2::3] = 0.05
        inits.append(p0)
        
    # Optimization loop
    for i, p0 in enumerate(inits):
        try:
            res = opt.minimize(objective, p0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            x_r, y_r, r_r = repair(res.x)
            s = np.sum(r_r)
            if s > best_sum:
                best_sum = s
                best_p = np.zeros(3 * N)
                best_p[0::3] = x_r
                best_p[1::3] = y_r
                best_p[2::3] = r_r
        except Exception:
            continue
            
        # Local perturbation phase around current best to escape local minima
        if best_p is not None:
            for _ in range(3):
                p_pert = best_p + np.random.normal(0, 0.0005, 3 * N)
                p_pert[0::3] = np.clip(p_pert[0::3], 0.01, 0.99)
                p_pert[1::3] = np.clip(p_pert[1::3], 0.01, 0.99)
                p_pert[2::3] = np.clip(p_pert[2::3], 1e-4, 0.45)
                try:
                    res = opt.minimize(objective, p_pert, method='SLSQP', bounds=bounds,
                                       constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    x_r, y_r, r_r = repair(res.x)
                    s = np.sum(r_r)
                    if s > best_sum:
                        best_sum = s
                        best_p[0::3] = x_r
                        best_p[1::3] = y_r
                        best_p[2::3] = r_r
                except Exception:
                    pass

    # Fallback
    if best_p is None:
        best_p = inits[0]
        
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3]
    return centers, radii, float(np.sum(radii))
