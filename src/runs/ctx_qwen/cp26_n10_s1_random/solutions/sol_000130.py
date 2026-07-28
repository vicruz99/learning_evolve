# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=ef04e2a0 sum of radii=2.401921 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def get_initial_configs(n=26):
    """Generates diverse initial configurations for optimization."""
    configs = []
    rng = np.random.default_rng(42)

    # 1. Hexagonal lattice
    pts = []
    y = 0.1
    row = 0
    while len(pts) < n:
        shift = 0.1 if row % 2 == 1 else 0.0
        x = 0.1 + shift
        while x + 0.1 <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 0.2
        y += np.sqrt(3) * 0.1
        row += 1
    configs.append(np.array(pts[:n]))

    # 2. Uniform grid
    g = np.linspace(0.15, 0.85, 6)
    pts2 = np.array([[x, y] for y in g for x in g])
    configs.append(pts2[:n])

    # 3. Controlled perturbations of hex lattice
    for _ in range(6):
        c = configs[0].copy() + rng.uniform(-0.03, 0.03, (n, 2))
        configs.append(np.clip(c, 0.05, 0.95))

    # 4. Random dense starts
    for _ in range(4):
        c = rng.uniform(0.2, 0.8, (n, 2))
        configs.append(c)

    return configs

def solve_lp_radii(centers):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Constraints: r_i >= 0, r_i <= dist_to_boundary, r_i + r_j <= dist(i,j)
    """
    n = centers.shape[0]
    
    # Boundary limits for each circle
    limits = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    bounds = [(0, lim) for lim in limits]

    # Objective: maximize sum(r) -> minimize -sum(r)
    c_obj = np.ones(n) * -1.0

    # Pairwise distance constraints
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)

    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def joint_objective(vars_arr, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[2::3])

def joint_constraints(vars_arr, n):
    """Computes inequality constraints >= 0 for valid packing."""
    xs = vars_arr[0::3]
    ys = vars_arr[1::3]
    rs = vars_arr[2::3]

    con = []
    # Boundary constraints
    con.extend(xs - rs)
    con.extend(1.0 - xs - rs)
    con.extend(ys - rs)
    con.extend(1.0 - ys - rs)

    # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    xs_m = xs[:, None] - xs[None, :]
    ys_m = ys[:, None] - ys[None, :]
    rs_m = rs[:, None] + rs[None, :]

    dist_sq = xs_m**2 + ys_m**2
    r_sum_sq = rs_m**2

    idx = np.triu_indices(n, k=1)
    con.extend(dist_sq[idx] - r_sum_sq[idx])

    return np.array(con)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    configs = get_initial_configs(n)

    best_sum = 0.0
    best_centers = None
    best_radii = None

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': joint_constraints, 'args': (n,)}

    for cfg in configs:
        v0 = np.zeros(3 * n)
        v0[0::3] = cfg[:, 0]
        v0[1::3] = cfg[:, 1]
        v0[2::3] = 0.09  # Feasible initial radius

        try:
            # Phase 1: Joint optimization of centers and radii
            res = minimize(joint_objective, v0, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})

            if np.isfinite(res.fun):
                cx = res.x[0::3]
                cy = res.x[1::3]
                centers_opt = np.column_stack((cx, cy))

                # Phase 2: Exact LP refinement for radii given optimized centers
                lp_radii, lp_sum = solve_lp_radii(centers_opt)
                
                if lp_radii is not None and lp_sum > best_sum:
                    # Strict validation matching grader tolerance
                    valid = True
                    for i in range(n):
                        x, y, r = centers_opt[i, 0], centers_opt[i, 1], lp_radii[i]
                        if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
                            valid = False
                            break
                    if valid:
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                                if d < lp_radii[i] + lp_radii[j] - 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        best_sum = lp_sum
                        best_centers = centers_opt.copy()
                        best_radii = lp_radii.copy()
        except Exception:
            continue

    # Fallback if optimization yields nothing
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        if best_radii is None:
            best_radii = np.full(n, 0.08)
            best_sum = np.sum(best_radii)

    # Safety scaling to guarantee strict validity within numerical tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
