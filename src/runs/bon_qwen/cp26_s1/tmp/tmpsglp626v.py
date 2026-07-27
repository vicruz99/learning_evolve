import numpy as np
from scipy.optimize import minimize


def objective(v, n):
    """Maximize sum of radii (minimize negative sum)."""
    s = 0.0
    for i in range(n):
        s += v[3 * i + 2]
    return -s


def constraint_xmin(v, idx):
    """x >= r (circle inside left boundary)."""
    return v[3 * idx] - v[3 * idx + 2]


def constraint_xmax(v, idx):
    """x + r <= 1 (circle inside right boundary)."""
    return 1.0 - v[3 * idx] - v[3 * idx + 2]


def constraint_ymin(v, idx):
    """y >= r (circle inside bottom boundary)."""
    return v[3 * idx + 1] - v[3 * idx + 2]


def constraint_ymax(v, idx):
    """y + r <= 1 (circle inside top boundary)."""
    return 1.0 - v[3 * idx + 1] - v[3 * idx + 2]


def constraint_overlap(v, i, j):
    """Distance between centers >= sum of radii."""
    dx = v[3 * i] - v[3 * j]
    dy = v[3 * i + 1] - v[3 * j + 1]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - v[3 * i + 2] - v[3 * j + 2]


def init_hexagonal(n):
    """Initialize circles in a hexagonal pattern."""
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.095)

    idx = 0
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            base_x = 0.095 + col * 0.17
            base_y = 0.095 + row * 0.148
            if row % 2 == 1:
                base_x += 0.085

            x = base_x
            y = base_y

            if x >= 0.095 and x <= 0.905 and y >= 0.095 and y <= 0.905:
                centers[idx] = [x, y]
                idx += 1

            col += 1
            if col > 8:
                break
        row += 1
        if row > 8:
            break

    return centers, radii


def compute_max_radii(centers, n, tol=1e-10):
    """Given fixed centers, compute maximum feasible radii."""
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = min(centers[i, 0], 1.0 - centers[i, 0],
                        centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = max(radii[i], 0.0)

    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sr = radii[i] + radii[j]
                if sr > dist + tol:
                    scale = max(0.0, (dist - tol) / sr)
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    return radii


def refine_positions(centers, radii, n, lr=0.005, max_iter=800):
    """Reposition centers to maximize space using repulsion forces."""
    c = centers.copy()
    r = radii.copy()

    for it in range(max_iter):
        grad = np.zeros((n, 2))

        for i in range(n):
            # Boundary forces
            margin_x = max(0.0, r[i] - c[i, 0])
            margin_y = max(0.0, r[i] - c[i, 1])
            margin_xr = max(0.0, c[i, 0] - (1.0 - r[i]))
            margin_yr = max(0.0, c[i, 1] - (1.0 - r[i]))

            grad[i, 0] += margin_x * 20 - margin_xr * 20
            grad[i, 1] += margin_y * 20 - margin_yr * 20

            # Repulsion from overlapping circles
            for j in range(n):
                if i == j:
                    continue
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                md = r[i] + r[j]
                if dist < md and dist > 1e-12:
                    strength = (md - dist) / dist * 8.0
                    grad[i, 0] += strength * dx
                    grad[i, 1] += strength * dy

        # Update positions
        c -= lr * grad
        lr *= 0.9993

        # Clamp to valid region
        for i in range(n):
            c[i, 0] = max(r[i], min(1.0 - r[i], c[i, 0]))
            c[i, 1] = max(r[i], min(1.0 - r[i], c[i, 1]))

    return c


def run_packing():
    n = 26

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for trial in range(25):
        np.random.seed(trial * 17 + 3)

        # Initialize with hexagonal pattern
        centers, radii = init_hexagonal(n)

        # Add random perturbation
        centers = centers + np.random.uniform(-0.015, 0.015, (n, 2))
        centers = np.clip(centers, 0.05, 0.95)

        # Build variable vector: [x0, y0, r0, x1, y1, r1, ...]
        v0 = np.zeros(3 * n)
        for i in range(n):
            v0[3 * i] = centers[i, 0]
            v0[3 * i + 1] = centers[i, 1]
            v0[3 * i + 2] = radii[i]

        # Bounds for each variable
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 0.5))

        # Build constraint list
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': constraint_xmin, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': constraint_xmax, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': constraint_ymin, 'args': (i,)})
            constraints.append({'type': 'ineq', 'fun': constraint_ymax, 'args': (i,)})

        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({'type': 'ineq', 'fun': constraint_overlap, 'args': (i, j)})

        # Run SLSQP optimization
        result = minimize(
            objective,
            v0,
            method='SLSQP',
            args=(n,),
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 6000, 'ftol': 1e-15, 'disp': False}
        )

        if result.success:
            c = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                c[i, 0] = result.x[3 * i]
                c[i, 1] = result.x[3 * i + 1]
                r[i] = result.x[3 * i + 2]

            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_centers = c.copy()
                best_radii = r.copy()

    # If optimization failed, use initial hexagonal
    if best_centers is None:
        best_centers, best_radii = init_hexagonal(n)
        best_sum = np.sum(best_radii)

    # Final alternating refinement
    for _ in range(3):
        best_radii = compute_max_radii(best_centers, n)
        best_centers = refine_positions(best_centers, best_radii, n)
        best_radii = compute_max_radii(best_centers, n)

    best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum