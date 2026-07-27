import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_score = -np.inf
    best_result = None

    # Objective function to maximize the minimum separation (d)
    # r = d / 2. We maximize d.
    def negative_min_separation(vars):
        centers = vars.reshape((n, 2))
        min_sep = np.inf
        
        # Boundary constraints: dist to edge >= r
        # dist_to_edge = min(x, 1-x, y, 1-y)
        # We want dist_to_edge >= r. Let r be a variable or derived?
        # Here we treat 'r' as a variable in the optimization, or we just maximize the min_sep 
        # and set r = min_sep / 2 at the end.
        # To simplify, we can define a function that returns the min separation between 
        # all pairs and boundaries.
        
        # Boundary distances
        dists_to_boundary = np.min(np.array([
            centers[:, 0],
            1 - centers[:, 0],
            centers[:, 1],
            1 - centers[:, 1]
        ]).T, axis=1)
        
        # Inter-circle distances
        dists_between = np.array([
            [np.linalg.norm(centers[i] - centers[j]) for j in range(n)] 
            for i in range(n)
        ])
        np.fill_diagonal(dists_between, np.inf)
        
        current_min = np.min(np.minimum(dists_to_boundary, np.min(dists_between, axis=1)))
        return -current_min # Negative because we minimize

    # Constraints to keep centers in [0, 1]
    bounds = [(0, 1)] * (2 * n)

    def optimize_from_start(start_points):
        nonlocal best_score, best_result
        result = minimize(
            negative_min_separation,
            start_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        current_min_sep = -result.fun
        if current_min_sep > best_score:
            best_score = current_min_sep
            best_result = result.x.reshape((n, 2))
            return True
        return False

    # --- Initial Heuristic Layouts ---

    # 1. Hexagonal-like grid (6x5 perturbation)
    np.random.seed(42)
    cols = 6
    rows = 5
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    
    init_centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 1) * dx
            y = (r + 1) * dy
            # Stagger odd rows
            if r % 2 == 1:
                x += dx / 2
            init_centers.append([x, y])
    
    # Add a 26th circle in a gap if needed (our loop above creates 30, take first 26)
    init_centers = init_centers[:n]
    optimize_from_start(np.array(init_centers))

    # 2. 5x5 grid with one added
    centers_5x5 = []
    step = 1.0 / 6.0
    for r in range(5):
        for c in range(5):
            centers_5x5.append([step * (c + 1), step * (r + 1)])
    # Add 26th near a hole
    centers_5x5.append([0.5, 0.5]) 
    optimize_from_start(np.array(centers_5x5[:n]))

    # 3. Random initialization
    for _ in range(5):
        random_centers = np.random.rand(n, 2) * 0.8 + 0.1
        optimize_from_start(random_centers)

    # 4. Spiral initialization
    theta = np.linspace(0, 10 * np.pi, n)
    r_spiral = np.linspace(0.2, 0.4, n)
    spiral_centers = np.column_stack([
        0.5 + r_spiral * np.cos(theta),
        0.5 + r_spiral * np.sin(theta)
    ])
    # Clip to [0.1, 0.9]
    spiral_centers = np.clip(spiral_centers, 0.1, 0.9)
    optimize_from_start(spiral_centers)

    # --- Final Calculation ---
    if best_result is not None:
        centers = best_result
        # Calculate max possible equal radius for this configuration
        dists_to_boundary = np.min(np.array([
            centers[:, 0],
            1 - centers[:, 0],
            centers[:, 1],
            1 - centers[:, 1]
        ]).T, axis=1)
        
        dists_between = np.array([
            [np.linalg.norm(centers[i] - centers[j]) for j in range(n)] 
            for i in range(n)
        ])
        np.fill_diagonal(dists_between, np.inf)
        
        min_dist = np.min(np.minimum(dists_to_boundary, np.min(dists_between, axis=1)))
        radii = np.full(n, min_dist / 2.0)
    else:
        # Fallback to simple grid if optimization failed
        centers = np.array([[0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2],
                            [0.2, 0.4], [0.4, 0.4], [0.6, 0.4], [0.8, 0.4],
                            [0.2, 0.6], [0.4, 0.6], [0.6, 0.6], [0.8, 0.6],
                            [0.2, 0.8], [0.4, 0.8], [0.6, 0.8], [0.8, 0.8]]).reshape(-1, 2)
        centers = centers[:n]
        radii = np.full(n, 0.05)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii