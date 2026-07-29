# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 62f34940) state=24320ea9 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x, n):
    """Negative sum of radii (minimized by solver)."""
    return -np.sum(x[2 * n :])

def constraints(x, n, pair_i, pair_j):
    """Returns array of constraint values (must be >= 0)."""
    cx = x[0 : 2 * n : 2]
    cy = x[1 : 2 * n : 2]
    r = x[2 * n :]

    c = []
    # Boundary constraints: circle must stay inside [0,1]x[0,1]
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)

    # Overlap constraints: squared distance >= (r_i + r_j)^2
    dx = cx[pair_i] - cx[pair_j]
    dy = cy[pair_i] - cy[pair_j]
    rsum = r[pair_i] + r[pair_j]
    c.append(dx ** 2 + dy ** 2 - rsum ** 2)

    return np.concatenate(c)

def make_feasible(x, n, pair_i, pair_j):
    """Adjusts radii to ensure the initial configuration strictly satisfies constraints."""
    cx = x[0 : 2 * n : 2]
    cy = x[1 : 2 * n : 2]
    r = x[2 * n :].copy()

    # Enforce boundary constraints
    r = np.minimum(r, cx)
    r = np.minimum(r, 1.0 - cx)
    r = np.minimum(r, cy)
    r = np.minimum(r, 1.0 - cy)

    # Iteratively resolve overlaps
    for _ in range(50):
        changed = False
        for idx in range(len(pair_i)):
            i, j = pair_i[idx], pair_j[idx]
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - 1e-9:
                excess = r[i] + r[j] - d
                r[i] -= excess / 2.0
                r[j] -= excess / 2.0
                changed = True
        if not changed:
            break

    r = np.maximum(r, 1e-6)
    x[2 * n :] = r
    return x

def get_init(n, seed, pair_i, pair_j, style="hex"):
    """Generates an initial configuration (hexagonal or random) and ensures feasibility."""
    rng = np.random.RandomState(seed)
    cx = np.zeros(n)
    cy = np.zeros(n)

    if style == "hex":
        idx = 0
        y = 0.08
        row = 0
        dy = 0.17
        dx = 0.17
        while idx < n:
            shift = 0.0 if row % 2 == 0 else dx / 2
            x_pos = 0.08 + shift
            while x_pos <= 0.92 and idx < n:
                cx[idx] = x_pos + rng.uniform(-0.01, 0.01)
                cy[idx] = y + rng.uniform(-0.01, 0.01)
                idx += 1
                x_pos += dx
            y += dy * np.sqrt(3) / 2
            row += 1
    else:
        cx = np.random.uniform(0.2, 0.8, n)
        cy = np.random.uniform(0.2, 0.8, n)

    cx = np.clip(cx, 0.05, 0.95)
    cy = np.clip(cy, 0.05, 0.95)
    r = np.full(n, 0.04)

    x0 = np.zeros(3 * n)
    x0[0 : 2 * n : 2] = cx
    x0[1 : 2 * n : 2] = cy
    x0[2 * n :] = r
    return make_feasible(x0, n, pair_i, pair_j)

def run_packing():
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n

    best_sum = 0.0
    best_x = None

    # Phase 1: Multiple restarts from diverse initial configurations
    for seed in range(40):
        x0 = get_init(n, seed, pair_i, pair_j, "hex" if seed % 2 == 0 else "rand")
        try:
            res = minimize(
                objective,
                x0,
                args=(n,),
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "args": (n, pair_i, pair_j),
                },
                options={"maxiter": 15000, "ftol": 1e-13, "disp": False},
            )
            if res.success:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Perturbation refinement to escape local minima
    if best_x is not None:
        for it in range(10):
            noise = np.random.randn(len(best_x)) * 0.002 * (0.9 ** it)
            x_curr = best_x + noise
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            x_curr = np.clip(x_curr, lb, ub)
            try:
                res = minimize(
                    objective,
                    x_curr,
                    args=(n,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints={
                        "type": "ineq",
                        "fun": constraints,
                        "args": (n, pair_i, pair_j),
                    },
                    options={"maxiter": 15000, "ftol": 1e-13, "disp": False},
                )
                if res.success:
                    s = -res.fun
                    if s > best_sum:
                        best_sum = s
                        best_x = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization fails completely
    if best_x is None:
        best_x = get_init(n, 0, pair_i, pair_j, "hex")

    centers = best_x[: 2 * n].reshape(n, 2)
    radii = best_x[2 * n :]

    # Phase 3: Strict post-processing to guarantee validation compliance
    radii = np.maximum(radii, 0.0)
    for i in range(n):
        mx = min(
            centers[i, 0], 1.0 - centers[i, 0],
            centers[i, 1], 1.0 - centers[i, 1]
        )
        radii[i] = min(radii[i], mx - 1e-9)

    # Iteratively fix any remaining micro-overlaps
    for _ in range(300):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(
                    centers[i, 0] - centers[j, 0],
                    centers[i, 1] - centers[j, 1]
                )
                if d < radii[i] + radii[j] - 1e-9:
                    exc = radii[i] + radii[j] - d
                    radii[i] -= exc / 2.0
                    radii[j] -= exc / 2.0
                    changed = True
        if not changed:
            break

    return centers, radii, float(np.sum(radii))
