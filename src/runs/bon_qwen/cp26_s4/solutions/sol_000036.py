# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e09efbf) state=0a7f5ef8 sum of radii=2.416983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize

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
    Given fixed centers, solve the LP to maximize sum of radii.
    Constraints:
    1. r_i >= 0
    2. r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i (Boundary)
    3. r_i + r_j <= dist(i, j) (Non-overlap)
    
    Returns radii array.
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints A_ub * r <= b_ub
    # We need to construct A_ub and b_ub
    
    # Number of constraints:
    # 4 * n (boundaries) + n*(n-1)/2 (pairwise)
    num_boundary = 4 * n
    num_pairs = n * (n - 1) // 2
    num_constraints = num_boundary + num_pairs
    
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        A_ub[idx, i] = 1.0
        b_ub[idx] = x
        idx += 1
        
        # r_i <= 1 - x
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - x
        idx += 1
        
        # r_i <= y
        A_ub[idx, i] = 1.0
        b_ub[idx] = y
        idx += 1
        
        # r_i <= 1 - y
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - y
        idx += 1
        
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            # r_i + r_j <= dist
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    # Bounds for r_i: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Use high performance solver if available, else default
    try:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: return zeros or boundary constrained
            # If LP fails, try to return a safe solution
            # This might happen if constraints are infeasible (centers too close to boundary or each other)
            # But centers are inside [0,1] and distinct?
            # If centers are identical, dist=0, r_i+r_j<=0 => r=0.
            return np.zeros(n)
    except Exception:
        return np.zeros(n)

def get_initial_centers(n):
    """
    Generate initial centers for n circles.
    Using a hexagonal grid pattern clipped to square, or a dense grid.
    """
    centers = []
    # Try to place points in a hexagonal lattice
    # Spacing roughly 1/sqrt(n) ?
    # For n=26, maybe 5x5 grid is good base
    
    # Let's try a slightly randomized grid to avoid symmetries
    rows = 6
    cols = 5 # 30 points, pick 26
    
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.1, 0.9, rows)
    
    points = []
    for r in range(rows):
        offset = 0.05 if r % 2 == 1 else 0.0 # Hex shift
        for c in range(cols):
            x = xs[c] + offset
            y = ys[r]
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # Shuffle and pick n
    np.random.shuffle(points)
    points = points[:n]
    
    return np.array(points)

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    np.random.seed(42) # For reproducibility
    
    # Initial centers
    centers = get_initial_centers(n)
    
    # Solve initial radii
    radii = solve_radii_lp(centers)
    best_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Simulated Annealing on centers
    # We perturb centers and re-solve LP for radii
    
    temp = 0.1 # Initial temperature
    alpha = 0.995 # Cooling rate
    steps = 2000 # Number of iterations
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    current_sum = best_sum
    
    # To ensure we don't get stuck, we might want to restart or use large moves
    # But let's try standard SA first
    
    for step in range(steps):
        # Generate perturbation
        # Randomly select a circle and move it, or move all slightly
        # Moving all slightly might be better for global search
        # Let's pick a random subset of circles to move
        
        move_mask = np.random.random(n) < 0.5 # 50% of circles move
        noise = np.random.normal(0, temp, size=(n, 2))
        
        new_centers = current_centers.copy()
        new_centers[move_mask] += noise[move_mask]
        
        # Project back to [0, 1] x [0, 1]
        # Actually, centers must be such that a circle can exist, so center must be in [r, 1-r].
        # But r is unknown. However, center must be in [0, 1].
        # If center is on boundary, r must be 0.
        # We can clip centers to [0, 1].
        new_centers = np.clip(new_centers, 0.0, 1.0)
        
        # Solve for radii
        new_radii = solve_radii_lp(new_centers)
        new_sum = np.sum(new_radii)
        
        # Acceptance criteria
        if new_sum > current_sum:
            delta = new_sum - current_sum
            # Accept
            current_centers = new_centers
            current_radii = new_radii
            current_sum = new_sum
            
            if new_sum > best_sum:
                best_sum = new_sum
                best_centers = new_centers.copy()
                best_radii = new_radii.copy()
        else:
            delta = new_sum - current_sum
            # Accept with probability exp(delta / temp)
            if delta < 0 and temp > 1e-9:
                prob = math.exp(delta / temp)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_radii = new_radii
                    current_sum = new_sum
        
        # Cool down
        temp *= alpha
        
    # Final validation
    is_valid = validate_packing(best_centers, best_radii)
    
    if not is_valid:
        # Fallback to a safe grid packing if optimization failed or produced invalid
        # This shouldn't happen if LP constraints are respected, but for safety
        # Construct a simple 5x5 grid + 1
        fallback_centers = []
        fallback_radii = []
        
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                fallback_centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                fallback_radii.append(0.1)
        
        # Add 26th circle in a gap?
        # Center of square (0.5, 0.5) is occupied? No, grid is 0.1, 0.3, 0.5...
        # (0.5, 0.5) is a center.
        # Try (0.1, 0.5) -> occupied.
        # Try mid-edge? (0.5, 0.05)?
        # Distance to (0.5, 0.1) is 0.05. Radius would be 0.
        # Let's just return the grid solution if needed, though it's lower sum.
        # But LP guarantees validity.
        pass

    # Return best solution
    return best_centers, best_radii, best_sum

# Helper to run and print
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
    # Print details
    for i in range(len(radii)):
        print(f"Circle {i}: center={centers[i]}, radius={radii[i]}")
