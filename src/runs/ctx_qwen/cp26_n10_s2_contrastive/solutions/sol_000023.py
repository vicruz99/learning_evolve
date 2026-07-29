# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state c2ddf6ac) state=b1d5d9d5 sum of radii=1.537272 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N_CIRCLES = 26

def solve_radii_lp(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(radii)
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    
    # Bounds for each r_i: [0, min_dist_to_boundary]
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, max_r)))
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists.append(math.sqrt(dx*dx + dy*dy))
            
    m = len(pairs)
    A_ub = np.zeros((m, n))
    for k, (i, j) in enumerate(pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
    b_ub = np.array(dists)
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Clamp tiny negative values from numerical noise
            radii = np.maximum(res.x, 0.0)
            return radii, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def obj_joint(params):
    """Objective function for joint optimization: maximize sum of radii."""
    n = N_CIRCLES
    return -np.sum(params[2*n:])

def con_joint(params):
    """
    Inequality constraints for joint optimization:
    - Boundary: r <= x <= 1-r, r <= y <= 1-r
    - Non-overlap: dist(i,j) >= r_i + r_j
    Returns array of constraint values (must be >= 0)
    """
    n = N_CIRCLES
    cx, cy, r = params[:n], params[n:2*n], params[2*n:]
    
    c = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    # Vectorized pairwise distances and radius sums
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    r_sum = r[:, None] + r[None, :]
    
    # Lower triangle mask for unique pairs
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    c_overlap = (dist - r_sum)[mask]
    
    return np.concatenate([c, c_overlap])

def run_packing():
    n = N_CIRCLES
    best_centers = None
    best_radii = None
    best_sum = 0.0

    cons = {'type': 'ineq', 'fun': con_joint}
    bounds_joint = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice patterns with various densities
    for row_mult in [0.85, 0.95, 1.05, 1.15]:
        for col_mult in [0.85, 0.95, 1.05]:
            c = []
            y = 0.06 * row_mult
            dy = 0.13 * row_mult
            for row in range(12):
                x_start = 0.06 * col_mult if row % 2 == 0 else 0.06 * col_mult + 0.065 * col_mult
                dx = 0.15 * col_mult
                col = 0
                while x_start + col * dx <= 0.94 and len(c) < n:
                    c.append([x_start + col * dx, y])
                    col += 1
                y += dy
                if y > 0.94: break
            if len(c) < n:
                while len(c) < n:
                    c.append([0.5, 0.5])
            inits.append(np.array(c[:n]))

    # 2. Uniform grid patterns
    for s in [4, 5, 6]:
        step = 1.0 / (s + 1)
        c = []
        for i in range(s):
            for j in range(s):
                if len(c) < n:
                    c.append([step * (i + 1), step * (j + 1)])
        while len(c) < n:
            c.append([0.5, 0.5])
        inits.append(np.array(c[:n]))

    # 3. Random dense packings
    for seed in range(20):
        np.random.seed(seed)
        c = np.random.uniform(0.15, 0.85, (n, 2))
        inits.append(c)

    # Phase 1: Joint optimization from diverse starts
    for c_init in inits:
        r_init = np.full(n, 0.04)
        x0 = np.concatenate([c_init.ravel(), r_init])
        
        try:
            res = minimize(obj_joint, x0, method='SLSQP', bounds=bounds_joint,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-14})
            if res.success:
                curr_centers = res.x[:2*n].reshape(n, 2)
                # LP refinement often finds strictly better radii for fixed centers
                opt_radii, lp_sum = solve_radii_lp(curr_centers)
                if lp_sum > best_sum:
                    best_sum = lp_sum
                    best_centers = curr_centers.copy()
                    best_radii = opt_radii.copy()
        except Exception:
            continue

    # Phase 2: Iterative local search around best solution
    if best_centers is not None:
        for trial in range(15):
            np.random.seed(trial * 7 + 3)
            # Perturb centers and radii slightly
            c_pert = best_centers + np.random.randn(n, 2) * 0.012
            c_pert = np.clip(c_pert, 0.02, 0.98)
            r_pert = best_radii + np.random.randn(n) * 0.006
            r_pert = np.clip(r_pert, 0.005, 0.5)
            x0 = np.concatenate([c_pert.ravel(), r_pert])
            
            try:
                res = minimize(obj_joint, x0, method='SLSQP', bounds=bounds_joint,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-15})
                if res.success:
                    curr_centers = res.x[:2*n].reshape(n, 2)
                    opt_radii, lp_sum = solve_radii_lp(curr_centers)
                    if lp_sum > best_sum:
                        best_sum = lp_sum
                        best_centers = curr_centers.copy()
                        best_radii = opt_radii.copy()
            except Exception:
                continue

    # Fallback safety net
    if best_centers is None:
        step = 0.2
        fallback_centers = np.array([[step*(i+1), step*(j+1)] for i in range(4) for j in range(4)])
        while len(fallback_centers) < n:
            fallback_centers = np.vstack([fallback_centers, [0.5, 0.5]])
        best_radii, best_sum = solve_radii_lp(fallback_centers)
        best_centers = fallback_centers

    return best_centers, best_radii, float(best_sum)
