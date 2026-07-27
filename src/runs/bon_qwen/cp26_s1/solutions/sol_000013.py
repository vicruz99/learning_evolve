# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=0b18d0a4 sum of radii=1.175352 correctness=1.0
# stdout(first 200): Sum of radii: 1.1753519685169085
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import time

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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n_circles = 26
    
    def create_hexagonal_seeds(rows_counts, r_init=0.05):
        centers = []
        y = r_init
        for count in rows_counts:
            row_width = (count - 1) * 2 * r_init
            x_start = (1.0 - row_width) / 2
            for i in range(count):
                x = x_start + i * 2 * r_init
                centers.append([x, y])
            y += r_init * np.sqrt(3)
        return np.array(centers)

    def create_perturbed_grid_seeds():
        # 5x5 grid with one extra
        centers = []
        # 5 rows, 5 cols
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + j * 0.2, 0.1 + i * 0.2])
        # Add 26th circle in a gap or random
        centers.append([0.5, 0.5]) # Center
        centers = np.array(centers)
        # Perturb slightly
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        return centers

    def create_random_seeds():
        return np.random.uniform(0.1, 0.9, (n_circles, 2))

    seeds = [
        create_hexagonal_seeds([5, 6, 5, 6, 4]),
        create_hexagonal_seeds([6, 5, 6, 5, 4]),
        create_hexagonal_seeds([5, 5, 5, 5, 6]),
        create_hexagonal_seeds([6, 6, 6, 5, 3]),
        create_hexagonal_seeds([5, 5, 5, 5, 5]), # 25 circles, add one random
        create_perturbed_grid_seeds(),
        create_random_seeds(),
        create_random_seeds(),
        create_random_seeds(),
    ]
    
    # Add one random circle to the 5x5 seed if needed (it has 26)
    # The 5x5 seed function above created 25 + 1 = 26.
    
    best_centers = None
    best_radii = None
    best_sum = 0.0

    for seed_idx, initial_centers in enumerate(seeds):
        if len(initial_centers) < n_circles:
            # Pad with random
            pad = np.random.uniform(0.1, 0.9, (n_circles - len(initial_centers), 2))
            initial_centers = np.vstack([initial_centers, pad])
        elif len(initial_centers) > n_circles:
            initial_centers = initial_centers[:n_circles]
            
        centers = initial_centers.copy()
        radii = np.full(n_circles, 0.05)
        
        # Optimization parameters
        lr = 0.01
        decay = 0.995
        num_iter = 500
        
        for step in range(num_iter):
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Circle-Circle repulsion
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Force to separate
                        if dist > 1e-9:
                            force_dir = diff / dist
                            force_mag = overlap * 0.5
                            forces[i] += force_dir * force_mag
                            forces[j] -= force_dir * force_mag
                        else:
                            # If centers coincide, push randomly
                            rand_dir = np.random.uniform(-1, 1, 2)
                            rand_dir /= np.linalg.norm(rand_dir)
                            forces[i] += rand_dir * 0.01
                            forces[j] -= rand_dir * 0.01
            
            # Wall forces (push away from boundaries)
            for i in range(n_circles):
                r = radii[i]
                x, y = centers[i]
                # Left
                if x - r < 0:
                    forces[i, 0] += (r - x) * 2.0
                # Right
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1) * 2.0
                # Bottom
                if y - r < 0:
                    forces[i, 1] += (r - y) * 2.0
                # Top
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1) * 2.0

            # Update centers
            centers += forces * lr
            
            # Keep centers in bounds roughly (soft constraint handled by forces, but clamp to prevent explosion)
            centers = np.clip(centers, 0.0, 1.0)
            
            # Expand radii slightly
            # Limit expansion to avoid huge overlaps immediately
            expansion_factor = 1.0 + 0.0005
            radii *= expansion_factor
            
            # Decay learning rate
            lr *= decay

        # Post-optimization: Shrink radii to fit exactly
        # Calculate max possible radius for each circle based on neighbors and walls
        # Then solve a small LP or just iterate to find consistent radii
        # For simplicity, we assume equal radii and find the max feasible r for this configuration
        
        # Find bottleneck distance
        min_sep = 1.0 # Start with wall distance
        for i in range(n_circles):
            x, y = centers[i]
            dist_wall = min(x, 1-x, y, 1-y)
            if dist_wall < min_sep:
                min_sep = dist_wall
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_sep:
                    min_sep = dist
        
        r_opt = min_sep / 2.0
        radii = np.full(n_circles, r_opt)
        
        current_sum = np.sum(radii)
        
        # Validate and update best
        # We need to ensure centers are valid for this r_opt
        # Since r_opt is derived from distances, it should be valid, 
        # but numerical errors might occur.
        # Let's validate.
        
        # Adjust centers slightly to ensure strict validity if needed?
        # Actually, r_opt is safe.
        
        if validate_packing(centers, radii):
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    return best_centers, best_radii, best_sum

# Run the packing
centers, radii, s = run_packing()
print(f"Sum of radii: {s}")
