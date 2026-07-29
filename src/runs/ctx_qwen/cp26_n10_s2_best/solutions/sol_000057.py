# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=f3e39bca sum of radii=2.614209 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def compute_constraints(v):
    """Inequality constraints: boundaries and non-overlap."""
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    # Boundary constraints
    c = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Pairwise non-overlap constraints (vectorized)
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum((c1 - c2)**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    c = np.concatenate([c, (dists - r_sum)[mask]])
    return c

def get_safe_radius(centers):
    """Compute a safe initial radius that guarantees feasibility."""
    n = centers.shape[0]
    min_d = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < min_d:
                min_d = d
        d = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if d < min_d:
            min_d = d
    return min_d / 2.0 * 0.90

def init_hex(seed):
    """Initialize with a hexagonal lattice pattern."""
    np.random.seed(seed)
    centers = []
    y = 0.12
    row = 0
    while len(centers) < N_CIRCLES:
        x_start = 0.12 if row % 2 == 0 else 0.28
        x = x_start
        while x <= 0.88 and len(centers) < N_CIRCLES:
            centers.append([x, y])
            x += 0.18
        y += 0.15
        row += 1
    centers = np.array(centers[:N_CIRCLES])
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    r = get_safe_radius(centers)
    return np.concatenate([centers.flatten(), np.full(N_CIRCLES, r)])

def init_grid(seed):
    """Initialize with a perturbed square grid."""
    np.random.seed(seed)
    centers = []
    for r in range(5):
        for c in range(6):
            if len(centers) >= N_CIRCLES:
                break
            x = 0.12 + c * 0.15
            y = 0.12 + r * 0.17
            centers.append([x, y])
    centers = np.array(centers[:N_CIRCLES])
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    r = get_safe_radius(centers)
    return np.concatenate([centers.flatten(), np.full(N_CIRCLES, r)])

def run_packing():
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    # Multi-start optimization
    for seed in range(15):
        for init_func in [init_hex, init_grid]:
            x0 = init_func(seed)
            try:
                res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': compute_constraints},
                               options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
                if res.success or np.all(compute_constraints(res.x) >= -1e-7):
                    current_sum = -res.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = res.x[:2*n].reshape(n, 2).copy()
                        best_radii = res.x[2*n:].copy()
            except Exception:
                pass

    if best_centers is None:
        best_centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        best_radii = np.full(n, 0.05)
        best_sum = np.sum(best_radii)

    centers = best_centers
    radii = best_radii
    
    # Strict enforcement of constraints to pass validation tolerance
    for _ in range(200):
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    viol = sum_r - dist
                    if viol > max_viol:
                        max_viol = viol
                    shrink = (viol + 1e-9) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
            x, y = centers[i]
            r_wall = min(x, 1.0 - x, y, 1.0 - y)
            if radii[i] > r_wall + 1e-12:
                viol = radii[i] - r_wall
                if viol > max_viol:
                    max_viol = viol
                radii[i] = r_wall
        if max_viol < 1e-13:
            break

    return centers, radii, float(np.sum(radii))
