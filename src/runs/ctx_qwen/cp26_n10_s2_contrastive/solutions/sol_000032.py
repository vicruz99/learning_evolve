# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 62f34940) state=25af70a4 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def objective(vars, n_circles):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2 * n_circles:])

def constraints(vars, n_circles, pairs):
    """
    Inequality constraints: g(vars) >= 0
    - Boundary: circles must stay inside [0,1]x[0,1]
    - Overlap: pairwise distances must be >= sum of radii
    """
    c = vars[:2 * n_circles].reshape(n_circles, 2)
    r = vars[2 * n_circles:]
    
    # Boundary constraints
    bc1 = c[:, 0] - r
    bc2 = 1.0 - c[:, 0] - r
    bc3 = c[:, 1] - r
    bc4 = 1.0 - c[:, 1] - r
    
    # Pairwise distance constraints
    dx = c[:, np.newaxis, 0] - c[np.newaxis, :, 0]
    dy = c[:, np.newaxis, 1] - c[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlap_cons = dists[pairs] - r_sum[pairs]
    
    return np.concatenate([bc1, bc2, bc3, bc4, overlap_cons])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Phase 1: Multiple restarts from hexagonal lattice seeds
    for seed in range(20):
        np.random.seed(seed)
        centers = np.zeros((n, 2))
        idx = 0
        row_counts = [6, 5, 6, 5, 4]
        y_step = 1.0 / (len(row_counts) + 1)
        
        for r_idx, count in enumerate(row_counts):
            y = (r_idx + 1) * y_step
            x_step = 1.0 / (count + 1)
            for c_idx in range(count):
                x = (c_idx + 1) * x_step
                if r_idx % 2 == 1:
                    x += x_step / 2.0
                x = np.clip(x + np.random.normal(0, 0.03), 0.05, 0.95)
                y = np.clip(y + np.random.normal(0, 0.03), 0.05, 0.95)
                centers[idx] = [x, y]
                idx += 1
                
        radii = np.full(n, 0.04)
        x0 = np.concatenate([centers.flatten(), radii])

        try:
            res = minimize(
                objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pairs)},
                options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False}
            )
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x[:2 * n].reshape(n, 2)
                    best_radii = res.x[2 * n:].copy()
        except Exception:
            continue

    # Phase 2: Basin hopping refinement from best found configuration
    if best_centers is not None:
        for _ in range(3):
            for trial in range(10):
                centers_pert = best_centers + np.random.normal(0, 0.005, best_centers.shape)
                radii_pert = best_radii + np.random.normal(0, 0.002, n)
                centers_pert = np.clip(centers_pert, 0.01, 0.99)
                radii_pert = np.clip(radii_pert, 0.001, 0.5)
                x0_pert = np.concatenate([centers_pert.flatten(), radii_pert])
                
                try:
                    res = minimize(
                        objective, x0_pert, args=(n,), method='SLSQP', bounds=bounds,
                        constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pairs)},
                        options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False}
                    )
                    if res.success:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_centers = res.x[:2 * n].reshape(n, 2)
                            best_radii = res.x[2 * n:].copy()
                except Exception:
                    continue

    # Phase 3: Deterministic post-processing to guarantee validity
    centers = best_centers.copy() if best_centers is not None else np.zeros((n, 2))
    radii = best_radii.copy() if best_radii is not None else np.zeros(n)

    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        radii[i] = min(radii[i], max_r - 1e-9)
        radii[i] = max(radii[i], 0.0)

    # Iteratively resolve remaining overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < radii[i] + radii[j] - 1e-9:
                    overlap = radii[i] + radii[j] - d
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break

    final_sum = np.sum(radii)
    return centers, radii, float(final_sum)
