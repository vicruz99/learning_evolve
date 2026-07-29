# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000086 (state e307a773) state=913458d9 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2 + 1e-16)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c = np.concatenate([c, dist - r_sum])
    
    return c

def solve_radii_lp(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    c_obj = -np.ones(N)
    n_constraints = 4*N + len(PAIR_I)
    A_ub = np.zeros((n_constraints, N))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for i in range(N):
        mx = min(centers[i, 0], 1.0 - centers[i, 0])
        my = min(centers[i, 1], 1.0 - centers[i, 1])
        bnd = min(mx, my)
        A_ub[idx, i] = 1.0
        b_ub[idx] = bnd
        idx += 1
        
    dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
    dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
    dists = np.hypot(dx, dy)
    
    for k in range(len(PAIR_I)):
        A_ub[idx + k, PAIR_I[k]] = 1.0
        A_ub[idx + k, PAIR_J[k]] = 1.0
        b_ub[idx + k] = dists[k]
        
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
    if res.success:
        return res.x
    # Fallback to safe small radii if LP fails
    return np.full(N, 0.05)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = []
    
    # 1. Hexagonal lattices with varying densities and shifts
    for r_est in [0.08, 0.09, 0.095, 0.10, 0.105]:
        for shift in np.linspace(-0.02, 0.02, 5):
            c = []
            y = r_est + shift
            row = 0
            while len(c) < N:
                x_start = r_est if row % 2 == 0 else 2 * r_est
                x = x_start
                while x <= 1.0 - r_est and len(c) < N:
                    c.append([x, y])
                    x += 2 * r_est
                y += np.sqrt(3) * r_est
                row += 1
            c = np.array(c[:N])
            c += np.random.uniform(-0.01, 0.01, c.shape)
            c = np.clip(c, 0.02, 0.98)
            inits.append(c)
            
    # 2. Random configurations
    for seed in range(15):
        np.random.seed(seed + 1000)
        inits.append(np.random.uniform(0.05, 0.95, (N, 2)))
        
    # Phase 1: Multi-start optimization
    for c_init in inits:
        # LP gives optimal radii for these centers, ensuring strict feasibility
        r_init = solve_radii_lp(c_init)
        v0 = np.concatenate([c_init[:, 0], c_init[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                if np.min(constraints(res.x)) >= -1e-6:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(20):
            np.random.seed(step + 2000)
            pert = current_v.copy()
            pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            pert[2*N:] *= 0.95  # Shrink to guarantee feasibility after perturbation
            
            # Re-solve LP to quickly adapt radii to new center positions
            centers_pert = pert[:2*N].reshape(N, 2)
            r_pert = solve_radii_lp(centers_pert)
            pert[2*N:] = r_pert * 0.99
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-6:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                pass
                
    # Fallback initialization
    if best_v is None:
        c_fb = np.random.uniform(0.1, 0.9, (N, 2))
        r_fb = solve_radii_lp(c_fb)
        best_v = np.concatenate([c_fb[:, 0], c_fb[:, 1], r_fb])
        
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(15):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
