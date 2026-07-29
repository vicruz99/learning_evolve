# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000041 (state 046a36a4) state=921aef56 sum of radii=2.633035 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I, J = np.triu_indices(N, k=1)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Squared distance and boundary constraints (must be >= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Squared distance avoids sqrt singularities and improves gradient behavior
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    c_overlap = dx**2 + dy**2 - (r[I] + r[J])**2
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_overlap, c_bound])

def solve_radii_lp(centers):
    """Given fixed centers, find radii that maximize sum via LP."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = np.zeros((n*(n-1)//2, n))
    b_ub = np.zeros(n*(n-1)//2)
    
    # Precompute pairwise distances
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :])**2).sum(axis=2))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    # Bounds for radii: 0 <= r_i <= distance to nearest wall
    bounds_r = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1-x, y, 1-y)
        bounds_r.append((0.0, max(0.0, ub)))
        
    # Try high-performance solver first, fallback to interior-point
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='interior-point')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.zeros(n), 0.0

def make_init_centers(seed, style='hex'):
    """Generate structured or random initial center configurations."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    if style == 'hex':
        # Hexagonal lattice approximation with varying spacing and offsets
        s = 0.14 + rng.uniform(0, 0.06)
        idx = 0
        row = 0
        y = s/2 + rng.uniform(0, 0.05)
        while idx < N and y < 1.0 - s/2:
            x_start = s/2 + (row % 2) * s/2 + rng.uniform(0, 0.02)
            col = 0
            while x_start + col*s < 1.0 - s/2 and idx < N:
                centers[idx, 0] = x_start + col*s
                centers[idx, 1] = y
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        # Fill any remaining slots randomly
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
    else:
        centers = rng.uniform(0.1, 0.9, (N, 2))
        
    # Add controlled noise to break symmetry
    centers += rng.normal(0, 0.008, centers.shape)
    return np.clip(centers, 0.05, 0.95)

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Broad multi-start search
    for seed in range(70):
        style = 'hex' if seed < 55 else 'random'
        c0 = make_init_centers(seed, style)
        r0 = np.full(N, 0.05)  # Feasible starting radii
        x0 = np.zeros(3*N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 12000, 'ftol': 1e-14, 'disp': False})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            curr_centers = np.column_stack((cx, cy))
            
            # Phase 2: Exact LP refinement for radii given optimized centers
            r_opt, s_opt = solve_radii_lp(curr_centers)
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_centers = curr_centers
                best_radii = r_opt
        except Exception:
            continue
            
    # Phase 3: Local perturbation refinement around the best solution
    if best_centers is not None:
        rng = np.random.RandomState(123)
        for _ in range(25):
            c_pert = best_centers + rng.normal(0, 0.003, best_centers.shape)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert, _ = solve_radii_lp(c_pert)
            
            x_pert = np.zeros(3*N)
            x_pert[0::3] = c_pert[:, 0]
            x_pert[1::3] = c_pert[:, 1]
            x_pert[2::3] = np.maximum(r_pert, 1e-5)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                cx = res.x[0::3]
                cy = res.x[1::3]
                curr_centers = np.column_stack((cx, cy))
                r_opt, s_opt = solve_radii_lp(curr_centers)
                
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = curr_centers
                    best_radii = r_opt
            except Exception:
                continue

    # Fallback safety net
    if best_centers is None:
        best_centers = make_init_centers(0)
        best_radii, best_sum = solve_radii_lp(best_centers)
        
    # Phase 4: Strict post-processing to guarantee validity
    radii = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        x, y = best_centers[i]
        max_r = min(x, 1-x, y, 1-y)
        if radii[i] > max_r - 1e-9:
            radii[i] = max(0.0, max_r - 1e-9)
            
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], 
                             best_centers[i,1]-best_centers[j,1])
                if d < radii[i] + radii[j] - 1e-9:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap/2.0
                    radii[j] -= overlap/2.0
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return best_centers, radii, float(np.sum(radii))
