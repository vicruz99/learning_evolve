import numpy as np
from scipy.optimize import minimize


def compute_overlap_energy(centers, radii):
    """Compute energy based on overlaps and boundary violations."""
    n = centers.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                energy += 100.0 * overlap ** 2
        # Boundary violations
        if centers[i, 0] - radii[i] < 0:
            energy += 100.0 * (centers[i, 0] - radii[i]) ** 2
        if centers[i, 0] + radii[i] > 1:
            energy += 100.0 * (centers[i, 0] + radii[i] - 1) ** 2
        if centers[i, 1] - radii[i] < 0:
            energy += 100.0 * (centers[i, 1] - radii[i]) ** 2
        if centers[i, 1] + radii[i] > 1:
            energy += 100.0 * (centers[i, 1] + radii[i] - 1) ** 2
    return energy


def apply_forces(centers, radii, dt=1e-4):
    """Apply repulsive forces to resolve overlaps and boundary issues."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)

    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff ** 2))
            min_dist = radii[i] + radii[j]

            if dist < min_dist and dist > 1e-12:
                overlap = min_dist - dist
                force = overlap * diff / dist
                forces[i] += force
                forces[j] -= force
            elif dist < 1e-12:
                force = min_dist * np.random.randn(2)
                forces[i] += force
                forces[j] -= force

        # Boundary forces
        if centers[i, 0] < radii[i]:
            forces[i, 0] += radii[i] - centers[i, 0]
        if centers[i, 0] > 1 - radii[i]:
            forces[i, 0] -= centers[i, 0] - (1 - radii[i])
        if centers[i, 1] < radii[i]:
            forces[i, 1] += radii[i] - centers[i, 1]
        if centers[i, 1] > 1 - radii[i]:
            forces[i, 1] -= centers[i, 1] - (1 - radii[i])

    centers += dt * forces

    # Clamp to feasible region
    for i in range(n):
        centers[i, 0] = max(radii[i], min(1 - radii[i], centers[i, 0]))
        centers[i, 1] = max(radii[i], min(1 - radii[i], centers[i, 1]))

    return centers


def get_min_clearance(centers, radii):
    """Get minimum clearance from any circle to others or boundaries."""
    n = centers.shape[0]
    min_clear = float('inf')

    for i in range(n):
        # Boundary clearance
        clear = min(
            centers[i, 0] - radii[i],
            1 - centers[i, 0] - radii[i],
            centers[i, 1] - radii[i],
            1 - centers[i, 1] - radii[i]
        )
        min_clear = min(min_clear, clear)

        # Neighbor clearance
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                clear = dist - radii[i] - radii[j]
                min_clear = min(min_clear, clear)

    return min_clear


def create_hexagonal_grid(n, r):
    """Create a hexagonal grid arrangement for n circles."""
    centers = []
    row_spacing = np.sqrt(3) * r

    for row in range(8):
        y = r + row * row_spacing
        if y + r > 1.0001:
            break
        for col in range(7):
            if row % 2 == 0:
                x = r + col * 2 * r
            else:
                x = 2 * r + col * 2 * r
            if x + r <= 1.0001 and len(centers) < n:
                centers.append([x, y])

    centers = np.array(centers[:n])
    radii = np.full(n, r)
    return centers, radii


def create_rectangular_grid(n, r):
    """Create a rectangular grid arrangement."""
    centers = []
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n:
                break
            x = r + col * 2 * r
            y = r + row * 2 * r
            if x + r <= 1.0001 and y + r <= 1.0001:
                centers.append([x, y])

    centers = np.array(centers[:n])
    radii = np.full(n, r)
    return centers, radii


def optimize_packing(centers, radii, n=26, max_iterations=30000):
    """Optimize packing using force-directed relaxation with gradual growth."""
    centers = centers.copy().astype(float)
    radii = radii.copy().astype(float)

    for iteration in range(max_iterations):
        # Determine step size (decreasing over time)
        if iteration < 10000:
            dt = 5e-4
        elif iteration < 20000:
            dt = 2e-4
        else:
            dt = 1e-4

        # Apply forces
        centers = apply_forces(centers, radii, dt)

        # Try to grow radii
        if iteration % 50 == 0:
            min_clear = get_min_clearance(centers, radii)
            if min_clear > 1e-8:
                grow = min(min_clear * 0.3, 5e-5)
                radii += grow

        # Add jitter to escape local minima
        if iteration % 2000 == 0 and iteration > 0:
            jitter = np.random.randn(*centers.shape) * 1e-4
            new_centers = centers + jitter
            # Validate new centers
            valid = True
            for i in range(n):
                if new_centers[i, 0] < radii[i] or new_centers[i, 0] > 1 - radii[i]:
                    valid = False
                    break
                if new_centers[i, 1] < radii[i] or new_centers[i, 1] > 1 - radii[i]:
                    valid = False
                    break
            if valid:
                centers = new_centers

    # Final cleanup
    for _ in range(5000):
        centers = apply_forces(centers, radii, 1e-6)

    return centers, radii


def run_packing():
    np.random.seed(42)
    n = 26

    best_sum = -1
    best_centers = None
    best_radii = None

    # Try multiple configurations
    configs = [
        create_hexagonal_grid(n, 0.08),
        create_hexagonal_grid(n, 0.085),
        create_hexagonal_grid(n, 0.09),
        create_rectangular_grid(n, 0.08),
    ]

    # Add perturbed versions
    for centers, radii in configs[:2]:
        for _ in range(3):
            jittered_centers = centers.copy() + np.random.randn(*centers.shape) * 0.01
            jittered_centers = np.clip(jittered_centers, 0.02, 0.98)
            configs.append((jittered_centers, radii.copy()))

    # Run optimization on each configuration
    for centers_init, radii_init in configs:
        centers, radii = optimize_packing(centers_init, radii_init, n, max_iterations=40000)
        current_sum = np.sum(radii)

        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Additional refinement with scipy optimization
    def objective(params):
        c = params[:2 * n].reshape(n, 2)
        r = params[2 * n:]
        return -np.sum(r) + compute_overlap_energy(c, r)

    params = np.concatenate([best_centers.flatten(), best_radii])

    bounds = []
    for _ in range(2 * n):
        bounds.append((0.001, 0.999))
    for _ in range(n):
        bounds.append((0.001, 0.5))

    result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000, 'ftol': 1e-12})

    refined_centers = result.x[:2 * n].reshape(n, 2)
    refined_radii = result.x[2 * n:]

    # Final force-based cleanup
    refined_centers, refined_radii = optimize_packing(refined_centers, refined_radii, n, max_iterations=10000)

    # Verify and fix any remaining violations
    for _ in range(1000):
        refined_centers = apply_forces(refined_centers, refined_radii, 1e-7)

    final_sum = np.sum(refined_radii)

    # Use whichever is better
    if final_sum > best_sum:
        best_sum = final_sum
        best_centers = refined_centers
        best_radii = refined_radii

    # Ensure valid output
    best_centers = np.array(best_centers, dtype=float)
    best_radii = np.array(best_radii, dtype=float)

    # Final sanity check - shrink if needed
    for _ in range(100):
        valid = True
        for i in range(n):
            if best_centers[i, 0] - best_radii[i] < -1e-12 or best_centers[i, 0] + best_radii[i] > 1 + 1e-12:
                valid = False
                break
            if best_centers[i, 1] - best_radii[i] < -1e-12 or best_centers[i, 1] + best_radii[i] > 1 + 1e-12:
                valid = False
                break
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                if dist < best_radii[i] + best_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        best_radii *= 0.999

    return best_centers, best_radii, np.sum(best_radii)