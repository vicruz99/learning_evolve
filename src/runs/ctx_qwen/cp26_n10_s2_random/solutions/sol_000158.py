# sol_000158 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000141 (state d8f6c168) state=a11c24e0 sum of radii=2.614876 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp_and_gradient(centers):
    """Solves LP for optimal radii given fixed centers and computes gradient via LP duals."""
    n = centers.shape[0]
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    A_ub = np.zeros((num_pairs + num_bound, n))
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pair_indices.append((i, j))
            idx += 1
    for i in range(n):
        for _ in range(4):
            A_ub[idx, i] = 1.0
            idx += 1

    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    b_ub = np.zeros(A_ub.shape[0])
    idx = 0
    for i, j in pair_indices:
        b_ub[idx] = dists[i, j]
        idx += 1
    for i in range(n):
        b_ub[idx] = centers[i, 0]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        b_ub[idx] = centers[i, 1]; idx += 1
        b_ub[idx] = 1.0 - centers[i, 1]; idx += 1

    c_obj = -np.ones(n)
    bounds_r = [(0, u) for u in ub]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    if not res.success:
        return None, None, None

    radii = res.x
    duals = res.ineqlin.marginals

    grad = np.zeros_like(centers)
    idx = 0
    for i, j in pair_indices:
        lam = duals[idx]
        if lam > 1e-7:
            d = dists[i, j]
            if d > 1e-9:
                vec = (centers[i] - centers[j]) / d
                grad[i] += lam * vec
                grad[j] -= lam * vec
        idx += 1

    boundary_start = len(pair_indices)
    for i in range(n):
        mu_L = duals[boundary_start + 4*i]
        mu_R = duals[boundary_start + 4*i + 1]
        mu_B = duals[boundary_start + 4*i + 2]
        mu_T = duals[boundary_start + 4*i + 3]
        grad[i, 0] += mu_L - mu_R
        grad[i, 1] += mu_B - mu_T

    return radii, -res.fun, grad

def optimize_gradient(centers0, max_iter=4000):
    """Runs gradient ascent on centers to maximize sum of radii."""
    centers = centers0.copy()
    best_centers = centers.copy()
    best_sum = -1.0
    step = 0.005
    no_improve = 0

    for k in range(max_iter):
        radii, curr_sum, grad = solve_lp_and_gradient(centers)
        if radii is None:
            break

        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > 60:
            step *= 0.75
        elif no_improve > 20:
            step *= 0.9

        if step < 1e-9:
            break

        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            centers += step * grad / g_norm

        centers = np.clip(centers, 0.005, 0.995)

        if k % 250 == 0 and k > 0:
            noise = 0.001 * (0.9**int(k/250))
            centers += np.random.normal(0, noise, centers.shape)
            centers = np.clip(centers, 0.005, 0.995)

    return best_centers, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)

    best_centers = None
    best_sum = -1.0

    # Phase 1: Generate diverse initial configurations
    starts = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5]
    ]

    for pat in patterns:
        for r_est in [0.095, 0.10, 0.105]:
            c = []
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = r_est if r_idx % 2 == 1 else 0.0
                x = r_est + shift
                for _ in range(cnt):
                    c.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3)
            c = np.array(c[:N])
            c += rng.normal(0, 0.002, c.shape)
            c = np.clip(c, 0.02, 0.98)
            starts.append(c)

    for _ in range(10):
        c = rng.uniform(0.15, 0.85, (N, 2))
        starts.append(c)

    # Phase 2: Gradient ascent from diverse starts
    for c0 in starts:
        c_opt, s_opt = optimize_gradient(c0)
        if s_opt > best_sum:
            best_sum = s_opt
            best_centers = c_opt.copy()

    # Phase 3: Equal-radius optimization phase
    # Exploits the fact that optimal packings are often near-equal radius
    def obj_eq(v): return -v[-1]
    def cons_eq(v):
        c = v[:2*N].reshape(N, 2)
        r = v[-1]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        idx = np.triu_indices(N, 1)
        d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
        con.append(d - 2*r)
        return np.concatenate(con)

    bounds_eq = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)]
    eq_starts = [best_centers] + starts[:5]
    for c0_eq in eq_starts:
        v0 = np.concatenate([c0_eq.flatten(), [0.095]])
        try:
            res_eq = minimize(obj_eq, v0, method='SLSQP', bounds=bounds_eq,
                              constraints={'type': 'ineq', 'fun': cons_eq},
                              options={'maxiter': 3000, 'ftol': 1e-13})
            s_eq = N * res_eq.x[-1]
            if s_eq > best_sum:
                best_sum = s_eq
                best_centers = res_eq.x[:2*N].reshape(N, 2)
        except Exception:
            pass

    # Phase 4: Second gradient ascent to relax radii equality
    c_opt2, s_opt2 = optimize_gradient(best_centers)
    if s_opt2 > best_sum:
        best_sum = s_opt2
        best_centers = c_opt2.copy()

    # Phase 5: Joint SLSQP polish on centers + radii
    def obj_joint(v): return -np.sum(v[2*N:])
    def cons_joint(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = [c[:,0]-r, 1.0-c[:,0]-r, c[:,1]-r, 1.0-c[:,1]-r]
        idx = np.triu_indices(N, 1)
        d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
        con.append(d - (r[idx[0]] + r[idx[1]]))
        return np.concatenate(con)

    r_lp, _, _ = solve_lp_and_gradient(best_centers)
    v0 = np.concatenate([best_centers.flatten(), r_lp])
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N

    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       options={'maxiter': 8000, 'ftol': 1e-14})
        if np.sum(res.x[2*N:]) > best_sum:
            best_centers = res.x[:2*N].reshape(N, 2)
            best_radii = res.x[2*N:]
        else:
            best_radii = r_lp
    except Exception:
        best_radii = r_lp

    # Phase 6: Strict numerical repair to guarantee validation compliance
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                req = best_radii[i] + best_radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
        for i in range(N):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            mr = min(x, 1.0-x, y, 1.0-y)
            if r > mr + 1e-12:
                best_radii[i] = mr
                changed = True
        if not changed:
            break

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(np.sum(best_radii))
