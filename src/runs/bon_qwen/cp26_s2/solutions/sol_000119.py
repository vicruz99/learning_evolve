# sol_000119 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=793a9d8f sum of radii=2.564071 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def get_initial_positions(n, seed):
    """Generate initial positions using a hexagonal lattice pattern."""
    np.random.seed(seed)
    pts = []
    s = 0.14  # Base spacing
    for i in range(10):
        for j in range(10):
            x = s * j + (i % 2) * s * 0.5
            y = s * i * np.sqrt(3) / 2
            if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                pts.append([x, y])
    np.random.shuffle(pts)
    return np.array(pts[:n])

def objective(v, n):
    """Objective function: maximize sum of radii => minimize negative sum."""
    s = 0.0
    for i in range(n):
        s += v[3 * i + 2]
    return -s

def constraints_fun(v, n):
    """Compute all inequality constraints g(v) >= 0."""
    c = []
    # Boundary & non-negativity
    for i in range(n):
        x, y, r = v[3 * i], v[3 * i + 1], v[3 * i + 2]
        c.append(x - r)          # x >= r
        c.append(1.0 - x - r)    # x + r <= 1
        c.append(y - r)          # y >= r
        c.append(1.0 - y - r)    # y + r <= 1
        c.append(r - 1e-6)       # r >= epsilon
    # Pairwise non-overlap
    for i in range(n):
        xi, yi, ri = v[3 * i], v[3 * i + 1], v[3 * i + 2]
        for j in range(i + 1, n):
            xj, yj, rj = v[3 * j], v[3 * j + 1], v[3 * j + 2]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            c.append(dist_sq - (ri + rj) ** 2)
    return np.array(c)

def run_packing():
    n = N_CIRCLES
    best_v = None
    best_val = 1e9  # Minimizing negative sum => more negative is better

    # Optimization bounds
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    constraint_def = {'type': 'ineq', 'fun': lambda v: constraints_fun(v, n)}

    # Multiple restarts to avoid local minima
    for seed in range(5):
        init_centers = get_initial_positions(n, seed)
        v0 = np.zeros(3 * n)
        for i in range(n):
            v0[3 * i] = init_centers[i, 0]
            v0[3 * i + 1] = init_centers[i, 1]
            v0[3 * i + 2] = 0.045  # Conservative initial radius

        try:
            res = minimize(
                objective, v0, args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_def,
                options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_v = res.x.copy()
        except Exception:
            continue

    if best_v is None:
        # Fallback to last run if something failed
        v0 = np.zeros(3 * n)
        for i in range(n):
            v0[3 * i] = 0.5
            v0[3 * i + 1] = 0.5
            v0[3 * i + 2] = 0.01
        best_v = v0

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = best_v[3 * i : 3 * i + 2]
        radii[i] = best_v[3 * i + 2]

    # Strict feasibility projection (handles numerical epsilon violations)
    safe = False
    iterations = 0
    while not safe and iterations < 50:
        safe = True
        iterations += 1
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Boundary constraints
            if x - r < 0: r = x
            if x + r > 1.0: r = 1.0 - x
            if y - r < 0: r = y
            if y + r > 1.0: r = 1.0 - y
            
            # Pairwise constraints
            for j in range(n):
                if i == j: continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if r + radii[j] > d:
                    r = d - radii[j]
            
            if r < radii[i] - 1e-8:
                safe = False
                radii[i] = max(0.0, r)
            else:
                radii[i] = r # Ensure consistency

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
