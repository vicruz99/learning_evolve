# sol_000229 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000188 (state 061cb89c) state=6807f97f sum of radii=2.603126 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers, n, i_idx, j_idx):
    """Solves LP to maximize sum of radii for fixed centers."""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    bounds = [(0.0, lim) for lim in limits]

    dx = centers[i_idx, 0] - centers[j_idx, 0]
    dy = centers[i_idx, 1] - centers[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    b_ub = dists

    m = len(i_idx)
    A_ub = np.zeros((m, n))
    for k, (u, v) in enumerate(zip(i_idx, j_idx)):
        A_ub[k, u] = 1.0
        A_ub[k, v] = 1.0

    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 1e-6 * n

def get_constraints(vars_arr, n, i_idx, j_idx):
    """Computes inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dr = rs[i_idx] + rs[j_idx]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs, dx**2 + dy**2 - dr**2])
    return c

def objective_func(vars_arr, n):
    """Objective: minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars_arr[2*n:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    rng = np.random.default_rng(42)

    i_idx, j_idx = np.triu_indices(n, k=1)
    bounds_vars = [(0.05, 0.95)] * (2 * n) + [(1e-5, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n, i_idx, j_idx)}

    # Generate diverse initial configurations
    configs = []
    row_dists = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 6, 5, 5], [6, 5, 5, 6, 4], [5, 6, 4, 6, 5],
        [4, 5, 6, 6, 5], [5, 4, 6, 6, 5], [6, 6, 6, 4, 4]
    ]

    for rd in row_dists:
        if sum(rd) < n: continue
        pts = []
        y = 0.12
        dy = 0.16
        for idx, cnt in enumerate(rd):
            shift = 0.10 if idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.19
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                if len(pts) < n:
                    pts.append([x_start + c * 0.19, y])
            y += dy
            if len(pts) >= n: break
        configs.append(np.array(pts[:n]))

    # Add random dense starts
    for _ in range(8):
        configs.append(rng.uniform(0.2, 0.8, (n, 2)))

    for cfg in configs:
        # Phase 1: Force-directed expansion to find jammed states
        centers_f = cfg.copy()
        radii_f = np.full(n, 0.02)
        for _ in range(300):
            radii_f += 0.0002
            forces = np.zeros_like(centers_f)
            # Boundary repulsion
            for d in range(2):
                viol1 = np.maximum(0, radii_f - centers_f[:, d])
                viol2 = np.maximum(0, centers_f[:, d] + radii_f - 1.0)
                forces[:, d] += (viol2 - viol1) * 200.0
            # Pairwise repulsion (vectorized)
            diff = centers_f[:, np.newaxis, :] - centers_f[np.newaxis, :, :]
            dists_mat = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dists_mat, np.inf)
            min_d = radii_f[:, np.newaxis] + radii_f[np.newaxis, :]
            overlap = np.maximum(0, min_d - dists_mat)
            dir_mat = diff / dists_mat[:, :, np.newaxis]
            dir_mat *= overlap[:, :, np.newaxis]
            forces += np.sum(dir_mat, axis=0) * 300.0

            centers_f += forces * 0.008
            centers_f = np.clip(centers_f, 0.02, 0.98)

        # Compute safe initial radius for SLSQP
        safe_r = np.min(np.minimum(np.minimum(centers_f[:,0], 1.0-centers_f[:,0]), 
                                   np.minimum(centers_f[:,1], 1.0-centers_f[:,1])))
        diffs = centers_f[:, np.newaxis, :] - centers_f[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        safe_r = min(safe_r, np.min(dists)/2.0)
        safe_r = max(safe_r, 0.02)

        v0 = np.zeros(3 * n)
        v0[:n] = centers_f[:, 0]
        v0[n:2*n] = centers_f[:, 1]
        v0[2*n:] = safe_r * 0.6

        try:
            # Phase 2: Joint SLSQP optimization
            res = minimize(objective_func, v0, args=(n,), method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                centers = np.column_stack((cx, cy))
                
                # Phase 3: Exact LP refinement
                r_lp, s_lp = solve_lp_radii(centers, n, i_idx, j_idx)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()

                    # Phase 4: Hill climbing on centers evaluated via LP
                    step = 0.008
                    for _ in range(300):
                        improved = False
                        for _ in range(15):
                            idx = rng.integers(n)
                            old_pos = best_centers[idx].copy()
                            best_centers[idx] += rng.uniform(-step, step, 2)
                            best_centers[idx] = np.clip(best_centers[idx], 0.05, 0.95)
                            r_try, s_try = solve_lp_radii(best_centers, n, i_idx, j_idx)
                            if s_try > best_sum + 1e-8:
                                best_sum = s_try
                                best_radii = r_try.copy()
                                improved = True
                            else:
                                best_centers[idx] = old_pos
                        if not improved:
                            step *= 0.8
                            if step < 1e-4:
                                break
        except Exception:
            continue

    # Fallback
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum = solve_lp_radii(best_centers, n, i_idx, j_idx)

    # Phase 5: Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    dx = best_centers[i_idx, 0] - best_centers[j_idx, 0]
    dy = best_centers[i_idx, 1] - best_centers[j_idx, 1]
    d = np.sqrt(dx**2 + dy**2)
    rs = best_radii[i_idx] + best_radii[j_idx]
    if np.any(rs > 1e-12):
        scale = min(scale, np.min(d / np.maximum(rs, 1e-12)))

    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, best_sum
