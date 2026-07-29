# sol_000119 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000107 (state 3d5163c6) state=ab7c4e6b sum of radii=2.633295 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_lp_radii(centers):
    """
    Given fixed centers, solve LP to maximize sum of radii.
    Constraints: r_i + r_j <= dist(i,j), r_i <= dist_to_boundary, r_i >= 0
    """
    n = centers.shape[0]
    ub = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_r = [(0, u) for u in ub]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_joint(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints_joint(v):
    """Computes all boundary and non-overlap constraints. Must be >= 0."""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    cons = []
    # Boundary constraints
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    # Overlap constraints
    idx = np.triu_indices(N, 1)
    d = np.linalg.norm(c[idx[0]] - c[idx[1]], axis=1)
    cons.append(d - (r[idx[0]] + r[idx[1]]))
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    np.random.seed(42)
    best_v = None
    best_sum = -1.0
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints_joint}
    
    starts = []
    
    # 1. Hexagonal lattice patterns with varying row structures
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 6, 6, 5],
        [6, 6, 5, 5, 4], [5, 5, 6, 5, 5], [4, 5, 6, 5, 6],
        [5, 5, 5, 6, 5], [5, 5, 4, 6, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6], [6, 6, 4, 5, 5], [5, 6, 4, 5, 6]
    ]
    
    for pat in patterns:
        for r_est in [0.09, 0.095, 0.10, 0.105]:
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
            c += np.random.normal(0, 0.005, c.shape)
            c = np.clip(c, 0.05, 0.95)
            r_init = np.full(N, r_est * 0.85)
            starts.append(np.concatenate([c.flatten(), r_init]))

    # 2. Random starts with pairwise repulsion
    for _ in range(20):
        c = np.random.uniform(0.1, 0.9, (N, 2))
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist < 0.18 and dist > 1e-6:
                        push = (0.18 - dist) * 0.5 / dist
                        c[i] += d_vec * push
                        c[j] -= d_vec * push
        c = np.clip(c, 0.05, 0.95)
        r_init = np.full(N, 0.07)
        starts.append(np.concatenate([c.flatten(), r_init]))

    # Phase 1: Multi-start constrained optimization + LP refinement
    for i, v0 in enumerate(starts):
        try:
            res = minimize(objective_joint, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
            if hasattr(res, 'x'):
                curr_sum = np.sum(res.x[2*N:])
                c_curr = res.x[:2*N].reshape(N, 2)
                r_lp, sum_lp = compute_lp_radii(c_curr)
                
                if sum_lp > curr_sum:
                    curr_sum = sum_lp
                    v_opt = np.concatenate([c_curr.flatten(), r_lp])
                else:
                    v_opt = res.x
                    
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = v_opt
        except Exception:
            pass

    # Fallback if optimization failed entirely
    if best_v is None:
        c0 = np.random.uniform(0.1, 0.9, (N, 2))
        r0, s0 = compute_lp_radii(c0)
        best_v = np.concatenate([c0.flatten(), r0])
        best_sum = s0

    # Phase 2: Adaptive perturbation search to escape local minima
    curr_v = best_v.copy()
    for step in range(40):
        noise = 0.005 * (0.82 ** step)
        v_pert = curr_v + np.random.normal(0, noise, curr_v.shape)
        v_pert = np.clip(v_pert, 0.0, 1.0)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.0, 0.5)
        
        try:
            res = minimize(objective_joint, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
            if hasattr(res, 'x'):
                c_curr = res.x[:2*N].reshape(N, 2)
                r_lp, sum_lp = compute_lp_radii(c_curr)
                curr_sum = max(np.sum(res.x[2*N:]), sum_lp)
                
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    if sum_lp >= np.sum(res.x[2*N:]):
                        best_v = np.concatenate([c_curr.flatten(), r_lp])
                    else:
                        best_v = res.x
                    curr_v = best_v
        except Exception:
            pass

    # Extract results
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict numerical repair to guarantee validation passes
    for _ in range(100):
        changed = False
        # Fix overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
