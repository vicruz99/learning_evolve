# sol_000028 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state 9fd6082b) state=2ad318ca sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective(vars):
    """Negative sum of radii (minimization)"""
    return -np.sum(vars[2 * N :])

def constraints(vars):
    """
    Inequality constraints g(vars) >= 0:
    - Boundary: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    - Non-overlap: dist(i,j) - r_i - r_j >= 0
    """
    cx = vars[:N]
    cy = vars[N : 2 * N]
    r = vars[2 * N :]
    
    n_cons = 4 * N + N * (N - 1) // 2
    c = np.empty(n_cons)
    
    c[:N] = cx - r
    c[N : 2 * N] = 1.0 - cx - r
    c[2 * N : 3 * N] = cy - r
    c[3 * N : 4 * N] = 1.0 - cy - r
    
    k = 4 * N
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            c[k] = np.sqrt(dx * dx + dy * dy) - r[i] - r[j]
            k += 1
    return c

def solve_lp_radii(centers):
    """
    Given fixed centers, solve LP to find radii that maximize sum(radii)
    subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Boundary constraints: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds_r = []
    for i in range(n):
        mx, my = centers[i]
        ub = min(mx, 1.0 - mx, my, 1.0 - my)
        bounds_r.append((0.0, max(0.0, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.zeros(n), 0.0

def gen_initial_params(seed):
    """Generate a valid initial configuration using a perturbed hexagonal lattice."""
    np.random.seed(seed)
    cx = np.zeros(N)
    cy = np.zeros(N)
    idx = 0
    row = 0
    
    # Hexagonal packing approximation
    while idx < N:
        y = 0.1 + row * 0.17 * np.sqrt(3) / 2
        x_start = 0.1 + (row % 2) * 0.085
        col = 0
        while x_start + col * 0.17 <= 0.9 and idx < N:
            cx[idx] = x_start + col * 0.17 + np.random.uniform(-0.02, 0.02)
            cy[idx] = y + np.random.uniform(-0.02, 0.02)
            idx += 1
            col += 1
        row += 1
        
    cx = np.clip(cx, 0.05, 0.95)
    cy = np.clip(cy, 0.05, 0.95)
    
    centers = np.column_stack((cx, cy))
    r, s = solve_lp_radii(centers)
    
    # Scale down slightly to ensure strict feasibility for SLSQP start
    r = r * 0.95
    return np.concatenate([cx, cy, r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a hybrid SLSQP + LP strategy with multiple restarts and local refinement.
    """
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Main optimization loop with diverse initializations
    for trial in range(35):
        x0 = gen_initial_params(trial)
        try:
            res = minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12}
            )
            
            curr_centers = np.column_stack((res.x[:N], res.x[N : 2 * N]))
            opt_r, opt_s = solve_lp_radii(curr_centers)
            
            if opt_s > best_sum:
                best_sum = opt_s
                best_centers = curr_centers
                best_radii = opt_r
                
                # Local refinement: perturb centers and re-optimize
                for _ in range(3):
                    pert = best_centers + np.random.randn(N, 2) * 0.002
                    pert = np.clip(pert, 0.01, 0.99)
                    r_p, s_p = solve_lp_radii(pert)
                    x0_p = np.concatenate([pert.ravel(), r_p * 0.99])
                    
                    res2 = minimize(
                        objective, x0_p, method='SLSQP', bounds=bounds,
                        constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12}
                    )
                    curr_centers2 = np.column_stack((res2.x[:N], res2.x[N : 2 * N]))
                    opt_r2, opt_s2 = solve_lp_radii(curr_centers2)
                    
                    if opt_s2 > best_sum:
                        best_sum = opt_s2
                        best_centers = curr_centers2
                        best_radii = opt_r2
                    else:
                        break
        except Exception:
            continue

    # Fallback if optimization fails (should not happen)
    if best_centers is None:
        centers = np.zeros((N, 2))
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < N:
                    centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                    idx += 1
        while idx < N:
            centers[idx] = [0.5, 0.5]
            idx += 1
        best_radii, best_sum = solve_lp_radii(centers)
        best_centers = centers

    return best_centers, best_radii, float(best_sum)
