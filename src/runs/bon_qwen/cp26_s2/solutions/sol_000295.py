# sol_000295 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=ffc2d560 sum of radii=1.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

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

def get_max_equal_radius(centers):
    """
    Calculates the maximum radius r such that all circles of radius r 
    centered at 'centers' fit in the unit square without overlapping.
    """
    # Wall constraints: r <= min(x, 1-x, y, 1-y) for all centers
    xs = centers[:, 0]
    ys = centers[:, 1]
    # Compute distances to 4 walls
    d_left = xs
    d_right = 1.0 - xs
    d_bottom = ys
    d_top = 1.0 - ys
    
    # The radius is limited by the minimum distance to any wall for any circle
    r_wall = np.min([d_left, d_right, d_bottom, d_top])
    
    # Pairwise constraints: 2*r <= distance(i, j)  =>  r <= distance(i, j) / 2
    # Compute pairwise distance matrix using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    
    # Radius is limited by half the minimum distance between any pair
    r_pairs = np.min(dists) / 2.0
    
    return min(r_wall, r_pairs)

def run_packing():
    N = 26
    
    # 1. Initialization
    # Generate a dense hexagonal grid with a small radius to ensure we have many points
    # to sample from. This ensures points are well-distributed initially.
    r_init = 0.04
    centers_list = []
    
    y = r_init
    row = 0
    while y <= 1 - r_init:
        if row % 2 == 0:
            x = r_init
            while x <= 1 - r_init:
                centers_list.append([x, y])
                x += 2 * r_init
        else:
            x = 2 * r_init
            while x <= 1 - r_init:
                centers_list.append([x, y])
                x += 2 * r_init
        y += math.sqrt(3) * r_init
        row += 1
    
    all_centers = np.array(centers_list)
    
    if len(all_centers) >= N:
        np.random.seed(42)
        # Randomly select N points to ensure good distribution across the square
        indices = np.random.choice(len(all_centers), N, replace=False)
        centers = all_centers[indices]
    else:
        np.random.seed(42)
        centers = np.random.uniform(r_init, 1-r_init, (N, 2))

    # 2. Optimization
    # Hill climbing to maximize the minimum radius (equal radii packing)
    # This effectively packs the circles as tightly as possible.
    
    best_r = get_max_equal_radius(centers)
    best_centers = centers.copy()
    
    current_centers = centers.copy()
    current_r = best_r
    
    np.random.seed(123)
    
    iterations = 15000
    step = 0.03
    decay = 0.9995
    
    for _ in range(iterations):
        idx = np.random.randint(0, N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * math.pi)
        magnitude = np.random.uniform(0, step)
        
        dx = math.cos(angle) * magnitude
        dy = math.sin(angle) * magnitude
        
        old_pos = current_centers[idx].copy()
        new_pos = old_pos + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        current_centers[idx] = new_pos
        
        # Evaluate new radius
        new_r = get_max_equal_radius(current_centers)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = current_centers.copy()
        else:
            # Revert move if it didn't improve the radius
            current_centers[idx] = old_pos
            
        step *= decay
        if step < 1e-6:
            break

    final_r = get_max_equal_radius(best_centers)
    radii = np.full(N, final_r)
    sum_radii = 26 * final_r
    
    # Validation safety check to handle potential numerical precision issues
    if not validate_packing(best_centers, radii):
        final_r *= 0.999
        radii = np.full(N, final_r)
        sum_radii = 26 * final_r

    return best_centers, radii, sum_radii
