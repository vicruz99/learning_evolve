# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state c2ddf6ac) state=42ee53fc sum of radii=2.587586 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N_CIRCLES = 26

def compute_overlap_constraints(centers, radii, n):
    """Compute non-overlap constraint values: dist(i,j) - r_i - r_j"""
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            cons.append(d - radii[i] - radii[j])
    return np.array(cons)

def solve_radii_lp(centers, n):
    """Given fixed centers, solve LP to find radii that maximize sum(radii)"""
    c_obj = -np.ones(n)
    
    # Upper bounds from square boundaries
    ub = np.array([min(c[0], 1 - c[0], c[1], 1 - c[1]) for c in centers])
    ub = np.maximum(ub, 1e-12)
    bounds = [(0, u) for u in ub]
    
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.hypot(dx, dy)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        
    if res.success:
        return res.x, -res.fun
    else:
        return np.zeros(n), 0.0

def objective_slsqp(x, n):
    """Objective for SLSQP: minimize negative sum of radii"""
    r = x[2 * n:]
    return -np.sum(r)

def constraints_slsqp(x, n):
    """All inequality constraints for SLSQP (must be >= 0)"""
    centers = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]
    con = []
    # Boundary constraints
    con.extend(centers[:, 0] - r)
    con.extend(1.0 - centers[:, 0] - r)
    con.extend(centers[:, 1] - r)
    con.extend(1.0 - centers[:, 1] - r)
    # Overlap constraints
    con.extend(compute_overlap_constraints(centers, r, n))
    return np.array(con)

def get_hex_initial(n, seed):
    """Generate a perturbed hexagonal lattice initialization"""
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))
    idx = 0
    dx = 0.16
    dy = dx * math.sqrt(3) / 2
    row = 0
    y = 0.08 + dy
    while idx < n:
        x_start = 0.08 + (row % 2) * (dx / 2)
        col = 0
        while True:
            x = x_start + col * dx
            if x > 0.92: break
            centers[idx] = [x + rng.uniform(-0.015, 0.015),
                            y + rng.uniform(-0.015, 0.015)]
            idx += 1
            col += 1
            if idx >= n: break
        y += dy
        row += 1
    return np.clip(centers, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    bounds_vars = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    for trial in range(20):
        # 1. Initialize with hexagonal pattern
        centers0 = get_hex_initial(n, trial)
        radii0, _ = solve_radii_lp(centers0, n)
        
        x0 = np.concatenate([centers0.flatten(), radii0])
        
        # 2. Optimize centers and radii jointly
        res = minimize(objective_slsqp, x0, args=(n,), method='SLSQP', bounds=bounds_vars,
                       constraints={'type': 'ineq', 'fun': constraints_slsqp, 'args': (n,)},
                       options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
        
        if res.success:
            c_opt = res.x[:2 * n].reshape(n, 2)
            r_lp, s_lp = solve_radii_lp(c_opt, n)
            
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_opt
                best_radii = r_lp
            
            # 3. Local search: perturb centers, solve LP, re-optimize if improved
            for _ in range(5):
                c_pert = c_opt + np.random.randn(n, 2) * 0.003
                c_pert = np.clip(c_pert, 0.005, 0.995)
                r_pert, s_pert = solve_radii_lp(c_pert, n)
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = c_pert
                    best_radii = r_pert
                    
                    x0_pert = np.concatenate([c_pert.flatten(), r_pert])
                    res2 = minimize(objective_slsqp, x0_pert, args=(n,), method='SLSQP', bounds=bounds_vars,
                                   constraints={'type': 'ineq', 'fun': constraints_slsqp, 'args': (n,)},
                                   options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False})
                    if res2.success:
                        c_opt2 = res2.x[:2 * n].reshape(n, 2)
                        r_lp2, s_lp2 = solve_radii_lp(c_opt2, n)
                        if s_lp2 > best_sum:
                            best_sum = s_lp2
                            best_centers = c_opt2
                            best_radii = r_lp2

    # 4. Final safety adjustment for numerical precision
    if best_centers is not None:
        centers = best_centers.copy()
        radii = best_radii.copy()
        
        for _ in range(10):
            changed = False
            # Fix boundary violations
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                max_r = min(x, 1 - x, y, 1 - y)
                if r > max_r - 1e-12:
                    radii[i] = max_r - 1e-12
                    changed = True
                    
            # Fix overlap violations
            for i in range(n):
                for j in range(i + 1, n):
                    d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                    if d < radii[i] + radii[j] - 1e-12:
                        overlap = radii[i] + radii[j] - d
                        radii[i] -= overlap / 2 + 1e-12
                        radii[j] -= overlap / 2 + 1e-12
                        changed = True
            if not changed:
                break
                
        best_centers = centers
        best_radii = np.maximum(radii, 1e-12)
        best_sum = np.sum(best_radii)
    else:
        # Fallback (should not trigger)
        best_centers = np.random.rand(n, 2) * 0.6 + 0.2
        best_radii = np.full(n, 0.02)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)
