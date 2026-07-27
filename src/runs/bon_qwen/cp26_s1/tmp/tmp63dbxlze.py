import numpy as np

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

def repulsion_optimize(n_circles=26, n_restarts=20, n_steps=1000):
    best_sum = 0
    best_centers = None
    best_radii = None

    for _ in range(n_restarts):
        # 1. Initialize with 5x5 grid + 1 seed circle
        # Grid centers for 25 circles
        grid_coords = []
        for i in range(5):
            for j in range(5):
                grid_coords.append([0.1 + i * 0.2, 0.1 + j * 0.2])
        grid_coords = np.array(grid_coords)
        
        # Seed for 26th circle (center of the square)
        centers = np.vstack((grid_coords, [[0.5, 0.5]]))
        
        # Perturb positions randomly
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        
        # Initialize radii (25 large, 1 small)
        radii = np.array([0.1] * 25 + [0.01])

        # 2. Repulsion Simulation
        learning_rate = 1e-4
        growth_rate = 5e-5
        stiffness = 10.0 # Power for repulsion force
        
        for step in range(n_steps):
            forces = np.zeros_like(centers)
            
            # Boundary repulsion
            for i in range(n_circles):
                x, y = centers[i]
                r = radii[i]
                
                # Push away from boundaries
                margin_x = 1.0 - (x + r)
                margin_y = 1.0 - (y + r)
                margin_neg_x = x - r
                margin_neg_y = y - r
                
                # Force proportional to distance from boundary (allows growth)
                # If circle is touching boundary, margin is 0. 
                # We want to increase r, so we simulate a force that increases r
                # and moves center away from wall.
                
                # Simple boundary constraint projection
                if x + r > 1.0:
                    forces[i, 0] -= (x + r - 1.0) * 100.0
                elif x - r < 0.0:
                    forces[i, 0] += (r - x) * 100.0
                
                if y + r > 1.0:
                    forces[i, 1] -= (y + r - 1.0) * 100.0
                elif y - r < 0.0:
                    forces[i, 1] += (r - y) * 100.0

            # Pairwise repulsion
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    
                    overlap = (radii[i] + radii[j]) - dist
                    if overlap > 0:
                        # Repulsive force
                        force_magnitude = stiffness * (overlap ** 2)
                        fx = (dx / dist) * force_magnitude
                        fy = (dy / dist) * force_magnitude
                        
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Update centers
            centers += learning_rate * forces
            
            # Update radii (grow if not overlapping too much, shrink if overlapping)
            # We can simply clamp radii growth based on distance to neighbors/boundaries
            # But a simpler heuristic for this optimization is:
            # If a circle is not significantly overlapping, it can grow.
            # We'll implement a specific growth step after position update.
            
            # Project centers to stay within valid range [r, 1-r] is hard as r changes.
            # Instead, ensure centers stay in [0,1] and then adjust r.
            centers[:, 0] = np.clip(centers[:, 0], 0, 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, 1)

            # Grow radii
            for i in range(n_circles):
                min_dist = 1.0
                # Check boundary
                min_dist = min(min_dist, centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
                
                # Check neighbors
                for j in range(n_circles):
                    if i == j: continue
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, (d - radii[j]))
                
                # Radius cannot exceed distance to nearest constraint
                # We add a small growth factor
                target_r = min_dist
                if target_r > radii[i]:
                    radii[i] += growth_rate * (target_r - radii[i])
                else:
                    radii[i] = target_r # Shrink if necessary
            
            # Decay learning rate to settle
            if step % 100 == 0:
                learning_rate *= 0.95
                growth_rate *= 0.99

        # Validation check for this restart
        if validate_packing(centers, radii):
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    return best_centers, best_radii, best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers, radii, sum_radii = repulsion_optimize()
    return centers, radii, sum_radii