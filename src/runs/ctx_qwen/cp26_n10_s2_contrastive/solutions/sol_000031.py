# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 62f34940) state=f74d293e sum of radii=0.451492 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math


def run_packing():
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = 0.0

    initializations = []
    
    # 1. Hexagonal lattice initializations
    for seed in range(10):
        centers = _make_hex_initial(n, seed)
        radii = _compute_feasible_radii(centers, n)
        initializations.append((centers, radii))
        
    # 2. Grid initializations
    for seed in range(5):
        centers = _make_grid_initial(n, seed)
        radii = _compute_feasible_radii(centers, n)
        initializations.append((centers, radii))
        
    # 3. Random dense initializations
    for seed in range(10):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, (n, 2))
        radii = _compute_feasible_radii(centers, n)
        initializations.append((centers, radii))

    # Optimize each initial configuration
    for idx, (c0, r0) in enumerate(initializations):
        c, r, s = _optimize_single(n, c0, r0)
        if c is not None and s > best_sum:
            best_sum = s
            best_centers = c.copy()
            best_radii = r.copy()

    # Refinement loop: perturb best solution and re-optimize with decreasing noise
    for iteration in range(8):
        rng = np.random.RandomState(iteration * 17 + 5)
        scale = 0.02 * (0.8 ** iteration)
        c_pert = best_centers + rng.uniform(-scale, scale, (n, 2))
        c_pert = np.clip(c_pert, 0.02, 0.98)
        r_pert = best_radii + rng.uniform(-scale / 2, scale / 2, n)
        r_pert = np.clip(r_pert, 0.001, 0.5)
        
        c, r, s = _optimize_single(n, c_pert, r_pert)
        if c is not None and s > best_sum:
            best_sum = s
            best_centers = c.copy()
            best_radii = r.copy()

    # Final validation and constraint fixing
    best_centers, best_radii = _fix_violations(best_centers, best_radii, n)
    best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)


def _make_hex_initial(n, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))
    idx = 0
    y = 0.08
    row = 0
    while idx < n:
        x = 0.08 if row % 2 == 0 else 0.14
        col = 0
        while x < 0.92 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 0.14
        y += 0.12
        row += 1
    centers += rng.uniform(-0.02, 0.02, (n, 2))
    return np.clip(centers, 0.03, 0.97)


def _make_grid_initial(n, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(6):
        for j in range(6):
            if idx < n:
                centers[idx] = [0.1 + i * 0.15 + rng.uniform(-0.01, 0.01),
                                0.1 + j * 0.15 + rng.uniform(-0.01, 0.01)]
                idx += 1
    return np.clip(centers, 0.03, 0.97)


def _compute_feasible_radii(centers, n):
    radii = np.zeros(n)
    for i in range(n):
        r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = math.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                r = min(r, d * 0.4)
        radii[i] = max(0.005, r)
    return radii


def _obj_func(x, n):
    return -np.sum(x[2 * n:])


def _constraints_func(x, n):
    cx = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]
    c_list = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_list.extend(cx[:, 0] - r)
    c_list.extend(1.0 - cx[:, 0] - r)
    c_list.extend(cx[:, 1] - r)
    c_list.extend(1.0 - cx[:, 1] - r)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i, 0] - cx[j, 0]
            dy = cx[i, 1] - cx[j, 1]
            c_list.append(dx * dx + dy * dy - (r[i] + r[j]) ** 2)
            
    return np.array(c_list)


def _optimize_single(n, centers, radii):
    x0 = np.concatenate([centers.ravel(), radii])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': _constraints_func, 'args': (n,)}
    
    try:
        res = minimize(
            _obj_func, x0, args=(n,), method='SLSQP', bounds=bounds,
            constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13, 'disp': False}
        )
        if res.success or -res.fun > 2.55:
            c = res.x[:2 * n].reshape(n, 2)
            r = res.x[2 * n:]
            return c, r, np.sum(r)
    except Exception:
        pass
    return None, None, 0.0


def _fix_violations(centers, radii, n):
    # Enforce boundary constraints strictly
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max_r - 1e-11)
        
    # Iteratively fix overlaps by shrinking radii proportionally
    for _ in range(100):
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    viol = req - d
                    radii[i] -= viol / 2.0
                    radii[j] -= viol / 2.0
                    max_viol = max(max_viol, viol)
        radii = np.maximum(radii, 0.0)
        if max_viol < 1e-12:
            break
    return centers, radii
