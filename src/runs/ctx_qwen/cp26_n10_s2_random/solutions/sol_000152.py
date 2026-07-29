# sol_000152 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000141 (state d8f6c168) state=16c0c8f4 sum of radii=2.614209 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIR_INDICES = np.triu_indices(N, k=1)

def obj_equal(v):
    return -v[-1]

def cons_equal(v):
    centers = v[:-1].reshape(N, 2)
    r = v[-1]
    c = []
    c.append(centers[:, 0] - r)
    c.append(1.0 - centers[:, 0] - r)
    c.append(centers[:, 1] - r)
    c.append(1.0 - centers[:, 1] - r)
    dx = centers[PAIR_INDICES[0], 0] - centers[PAIR_INDICES[1], 0]
    dy = centers[PAIR_INDICES[0], 1] - centers[PAIR_INDICES[1], 1]
    c.append(dx**2 + dy**2 - 4.0 * r**2)
    return np.concatenate(c)

def obj_joint(v):
    return -np.sum(v[2*N:])

def cons_joint(v):
    centers = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    c = []
    c.append(centers[:, 0] - r)
    c.append(1.0 - centers[:, 0] - r)
    c.append(centers[:, 1] - r)
    c.append(1.0 - centers[:, 1] - r)
    dx = centers[PAIR_INDICES[0], 0] - centers[PAIR_INDICES[1], 0]
    dy = centers[PAIR_INDICES[0], 1] - centers[PAIR_INDICES[1], 1]
    dr = r[PAIR_INDICES[0]] + r[PAIR_INDICES[1]]
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def solve_lp_radii(centers):
    n = centers.shape[0]
    ub = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    ub = np.maximum(ub, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    num_pairs = N * (N - 1) // 2
    num_bound = 4 * N
    A_ub = np.zeros((num_pairs + num_bound, n))
    b_ub = np.zeros(num_pairs + num_bound)
    
    idx = 0
    for i, j in zip(PAIR_INDICES[0], PAIR_INDICES[1]):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dists[i, j]
        idx += 1
        
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 0]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = centers[i, 1]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - centers[i, 1]; idx += 1
        
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, u) for u in ub], method='highs')
    return res.x if res.success else ub * 0.5

def hex_init(r_est, rows, rng):
    centers = []
    y = r_est
    for r_idx, cnt in enumerate(rows):
        shift = r_est if r_idx % 2 == 1 else 0.0
        x = r_est + shift
        for _ in range(cnt):
            centers.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    rng = np.random.default_rng(42)

    best_v_eq = None
    best_r_eq = 0.0

    patterns = [[5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [6,4,6,5,5], [4,6,6,6,4], [5,5,5,5,6]]
    r_starts = [0.092, 0.098, 0.102, 0.105]

    # Phase 1: Optimize layout for equal radii
    bounds_eq = [(0, 1)] * (2 * N) + [(0.05, 0.2)]
    for pat in patterns:
        for r0 in r_starts:
            for _ in range(3):
                c0 = hex_init(r0, pat, rng)
                v0 = np.concatenate([c0.flatten(), [r0]])
                try:
                    res = minimize(obj_equal, v0, method='SLSQP', bounds=bounds_eq,
                                   constraints={'type': 'ineq', 'fun': cons_equal},
                                   options={'maxiter': 3000, 'ftol': 1e-13})
                    r_val = -res.fun
                    if np.all(cons_equal(res.x) >= -1e-5) and r_val > best_r_eq:
                        best_r_eq = r_val
                        best_v_eq = res.x.copy()
                except Exception:
                    pass

    if best_v_eq is None:
        c0 = hex_init(0.095, [5,6,5,6,4], rng)
        best_v_eq = np.concatenate([c0.flatten(), [0.095]])
        best_r_eq = 0.095

    # Phase 2: Joint optimization with variable radii
    c_eq = best_v_eq[:2*N].reshape(N, 2)
    v0_joint = np.concatenate([c_eq.flatten(), np.full(N, best_r_eq * 0.99)])
    bounds_joint = [(0, 1)] * (2 * N) + [(0.0, 0.5)] * N

    best_v_joint = None
    best_sum_joint = -1.0

    # Multiple perturbed starts to escape local minima
    for trial in range(15):
        if trial > 0:
            v_trial = best_v_eq.copy()
            v_trial[:2*N] += rng.normal(0, 0.0008 * (1.0 + trial*0.2), 2*N)
            v_trial = np.clip(v_trial, 0.01, 0.99)
            v0_j = np.concatenate([v_trial[:2*N], np.full(N, v_trial[-1] * 0.99)])
        else:
            v0_j = v0_joint

        try:
            res = minimize(obj_joint, v0_j, method='SLSQP', bounds=bounds_joint,
                           constraints={'type': 'ineq', 'fun': cons_joint},
                           options={'maxiter': 5000, 'ftol': 1e-13})
            s_val = -res.fun
            if np.all(cons_joint(res.x) >= -1e-5) and s_val > best_sum_joint:
                best_sum_joint = s_val
                best_v_joint = res.x.copy()
        except Exception:
            pass

    if best_v_joint is None:
        best_v_joint = v0_joint
        best_sum_joint = best_r_eq * N

    centers = best_v_joint[:2*N].reshape(N, 2)
    
    # Phase 3: LP refinement for fixed centers to squeeze max radii
    radii = solve_lp_radii(centers)
    best_sum_joint = np.sum(radii)

    # Phase 4: Strict numerical repair to guarantee validation passes
    for _ in range(100):
        changed = False
        for i in range(N):
            x, y = centers[i]
            mr = min(x, 1.0 - x, y, 1.0 - y)
            if radii[i] > mr + 1e-9:
                radii[i] = mr
                changed = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-9:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break

    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
