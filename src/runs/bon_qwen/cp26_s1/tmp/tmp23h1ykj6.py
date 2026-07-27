import numpy as np
import cvxpy as cp
import math


def solve_lp(centers, n):
    """Solve LP to find optimal radii for given centers."""
    r = cp.Variable(n)
    constraints = [r >= 0]

    for i in range(n):
        constraints += [r[i] <= centers[i, 0]]
        constraints += [r[i] <= 1 - centers[i, 0]]
        constraints += [r[i] <= centers[i, 1]]
        constraints += [r[i] <= 1 - centers[i, 1]]

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            constraints += [r[i] + r[j] <= dist]

    prob = cp.Problem(cp.Maximize(cp.sum(r)), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return np.zeros(n)
    return np.array(r.value)


def adjust_centers(centers, radii, n, step_size=0.1):
    """Adjust centers using force-directed method based on tight constraints."""
    forces = np.zeros_like(centers)
    eps = 1e-12

    # Pairwise repulsion for tight/overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            required = radii[i] + radii[j]

            if dist < required:
                # Overlapping: strong repulsion
                if dist > eps:
                    repulsion = (required - dist) * step_size
                    fx = (dx / dist) * repulsion
                    fy = (dy / dist) * repulsion
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                else:
                    # Circles on top of each other, push randomly
                    angle = 2 * math.pi * (i + j * 7) / (n * (n - 1) / 2)
                    forces[i, 0] += math.cos(angle) * step_size * 0.1
                    forces[i, 1] += math.sin(angle) * step_size * 0.1
                    forces[j, 0] -= math.cos(angle) * step_size * 0.1
                    forces[j, 1] -= math.sin(angle) * step_size * 0.1

    # Boundary forces: push circles away from tight boundaries
    for i in range(n):
        for dim in range(2):
            # Left/bottom boundary
            slack = centers[i, dim] - radii[i]
            if slack < 0:
                forces[i, dim] += (-slack) * step_size * 2
            # Right/top boundary
            slack = (1 - centers[i, dim]) - radii[i]
            if slack < 0:
                forces[i, dim] -= (-slack) * step_size * 2

    return centers + forces


def clamp_to_square(centers, radii, n):
    """Ensure all circles are inside the unit square."""
    for i in range(n):
        centers[i, 0] = max(radii[i], min(1 - radii[i], centers[i, 0]))
        centers[i, 1] = max(radii[i], min(1 - radii[i], centers[i, 1]))
    return centers


def initialize_hexagonal(n, cols=5, rows=5):
    """Initialize circles in a hexagonal lattice pattern."""
    centers = []
    y = 0.1
    for row in range(rows):
        if row % 2 == 0:
            num_in_row = cols
            x_start = 0.1
        else:
            num_in_row = cols - 1 if cols > 1 else cols
            x_start = 0.1 + 0.1 * math.sqrt(3) / 2
        for col in range(num_in_row):
            x = x_start + col * 0.2
            if len(centers) < n:
                centers.append([x, y])
        y += 0.1 * math.sqrt(3) / 2

    centers = np.array(centers[:n])
    return centers


def initialize_grid(n):
    """Initialize circles in a grid pattern."""
    centers = []
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n:
                break
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            centers.append([x, y])
        if len(centers) >= n:
            break
    return np.array(centers[:n])


def initialize_mixed(n):
    """Initialize with a mix of hexagonal patterns."""
    centers = []
    # Try a denser hexagonal arrangement
    r_init = 0.09
    y = r_init
    for row in range(6):
        if row % 2 == 0:
            x = r_init
            num_in_row = int((1 - 2 * r_init) / (2 * r_init)) + 1
        else:
            x = r_init + r_init * math.sqrt(3) / 2
            num_in_row = int((1 - 2 * r_init - r_init * math.sqrt(3) / 2) / (2 * r_init)) + 1
        
        for col in range(num_in_row):
            if len(centers) < n:
                centers.append([x + col * 2 * r_init, y])
        y += r_init * math.sqrt(3)

    centers = np.array(centers[:n])
    return centers


def run_packing():
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None

    # Try multiple initial configurations
    initializers = [
        lambda: initialize_hexagonal(n, 5, 6),
        lambda: initialize_hexagonal(n, 6, 5),
        lambda: initialize_grid(n),
        lambda: initialize_mixed(n),
    ]

    for init_func in initializers:
        centers = init_func()
        
        # Ensure valid initial positions
        centers[:, 0] = np.clip(centers[:, 0], 0.01, 0.99)
        centers[:, 1] = np.clip(centers[:, 1], 0.01, 0.99)
        
        # Iterative optimization with decreasing step size
        radii = np.ones(n) * 0.05
        
        for iteration in range(2000):
            step_size = 0.05 * (1 - iteration / 2000) + 0.001
            
            radii = solve_lp(centers, n)
            current_sum = np.sum(radii)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
            
            if iteration < 1500:
                centers = adjust_centers(centers, radii, n, step_size=step_size)
                centers = clamp_to_square(centers, radii, n)
            
            # Add small random perturbation in later iterations
            if iteration > 1000 and iteration % 50 == 0:
                perturbation = np.random.randn(n, 2) * 0.005
                centers += perturbation
                centers[:, 0] = np.clip(centers[:, 0], 0.001, 0.999)
                centers[:, 1] = np.clip(centers[:, 1], 0.001, 0.999)

    # Final LP solve to ensure optimal radii for best centers
    final_radii = solve_lp(best_centers, n)
    
    # Verify and clamp
    for i in range(n):
        best_centers[i, 0] = max(final_radii[i], min(1 - final_radii[i], best_centers[i, 0]))
        best_centers[i, 1] = max(final_radii[i], min(1 - final_radii[i], best_centers[i, 1]))
    
    final_radii = solve_lp(best_centers, n)
    
    # Ensure no NaN
    final_radii = np.nan_to_num(final_radii, nan=0.01)
    best_centers = np.nan_to_num(best_centers, nan=0.5)
    
    return best_centers, final_radii, np.sum(final_radii)