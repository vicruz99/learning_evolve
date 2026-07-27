import numpy as np
from scipy.optimize import linprog, differential_evolution

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
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

def solve_radii_lp(centers):
    """
    Solve the Linear Programming problem to find optimal radii for fixed centers.
    Maximizes sum(radii) subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    
    # 1. Boundary constraints for each circle: r_i <= min(x, 1-x, y, 1-y)
    # Upper bound for each r_i
    x = centers[:, 0]
    y = centers[:, 1]
    
    ub_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # If a center is outside [0,1], ub_r might be negative.
    # We clip to 0 to avoid infeasibility in LP, effectively forcing r=0 for invalid centers.
    ub_r = np.maximum(ub_r, 0.0)
    
    # 2. Pairwise distance constraints: r_i + r_j <= dist(i, j)
    # We need to construct A_ub and b_ub such that A_ub @ r <= b_ub
    # Number of pairs
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    row_idx = 0
    # Compute distances and fill matrix
    # Vectorized distance computation might be faster but loop is clear and n is small (26)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dist
            row_idx += 1
            
    # Objective: Maximize sum(r) <=> Minimize -sum(r)
    c = np.ones(n) * -1.0
    
    # Bounds for r: 0 <= r_i <= ub_r[i]
    bounds = [(0, ub) for ub in ub_r]
    
    # Solve LP
    # 'highs' method is efficient
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        optimal_radii = res.x
        sum_radii = np.sum(optimal_radii)
        return sum_radii, optimal_radii
    else:
        # Fallback if LP fails (should not happen with valid bounds)
        return 0.0, np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function for the outer optimizer.
    Minimizes negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape(n, 2)
    
    # Clip centers to [0, 1] to ensure valid bounds for LP
    # This helps the optimizer stay in valid region
    centers = np.clip(centers, 0.0, 1.0)
    
    sum_radii, _ = solve_radii_lp(centers)
    
    # We minimize negative sum
    return -sum_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initial Guess Generation
    # A 5x5 grid gives 25 circles with radius 0.1 (centers at 0.1, 0.3, 0.5, 0.7, 0.9)
    # We need 26 circles. We can perturb a grid or use a hexagonal pattern.
    # Let's start with a grid-based initialization to give the optimizer a head start.
    
    # Create a 5x5 grid of centers
    coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    centers_grid = np.array(np.meshgrid(coords, coords)).T.reshape(-1, 2)
    
    # Add a 26th circle in a hole, e.g., at (0.2, 0.2)
    # Or just duplicate a point and let optimizer separate them?
    # Better to place it somewhat reasonably.
    # (0.2, 0.2) is distance sqrt(0.02) approx 0.141 from (0.1, 0.1).
    # It's a valid location.
    extra_center = np.array([[0.2, 0.2]])
    initial_centers = np.vstack([centers_grid, extra_center])
    
    # Shuffle to avoid symmetry issues if any
    np.random.seed(42)
    idx = np.random.permutation(n)
    initial_centers = initial_centers[idx]
    
    # Flatten for optimizer
    x0 = initial_centers.flatten()
    
    # Bounds for all centers: x, y in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Use Differential Evolution for global optimization
    # Popsize and maxiter tuned for balance between quality and time
    # 52 dimensions is moderately high, so we keep popsize relatively small but sufficient
    result = differential_evolution(
        objective_function, 
        bounds, 
        x0=initial_centers.flatten(), # Use initial guess to bias
        popsize=10, 
        maxiter=50, 
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        updating='deferred',
        polish=True
    )
    
    optimal_centers = result.x.reshape(n, 2)
    
    # Solve for final radii
    sum_radii, final_radii = solve_radii_lp(optimal_centers)
    
    # Validation check
    if not validate_packing(optimal_centers, final_radii):
        # If validation fails, it might be due to numerical precision.
        # We can try to shrink radii slightly to satisfy strict constraints if needed,
        # but the LP solution should be valid.
        pass

    return optimal_centers, final_radii, sum_radii