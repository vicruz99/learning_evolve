# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 12653929) state=2ca85051 sum of radii=0.000000 correctness=1.0
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # 1. Initialization: Staggered Hexagonal-like Grid
    centers = np.zeros((n, 2))
    r_start = 0.02
    r = r_start
    idx = 0
    
    rows = 6
    cols_per_row = [5, 4, 5, 4, 5, 3] # Total 26
    x_spacing = 1.0 / 6.0
    y_spacing = math.sqrt(3)/2 * x_spacing
    
    current_y = r + y_spacing
    for row_idx, cols in enumerate(cols_per_row):
        current_x = r + (5 - cols) * x_spacing / 2
        for c in range(cols):
            centers[idx, 0] = current_x
            centers[idx, 1] = current_y
            current_x += x_spacing
            idx += 1
        current_y += y_spacing
        
    radii = np.full(n, r)

    # 2. Physics-based Expansion and Relaxation
    expansion_rate = 0.0005
    cooling_rate = 0.9995
    repulsion_strength = 1.0
    iterations = 5000

    for step in range(iterations):
        # Grow radius
        r += expansion_rate
        radii[:] = r
        expansion_rate *= cooling_rate
        
        # Relaxation (simulate forces)
        forces = np.zeros_like(centers)
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 10000
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10000
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 10000
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10000
        
        # Inter-circle repulsion
        # Vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask to ignore diagonal and very close points
        mask = np.eye(n, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Calculate overlap amounts
        overlaps = 2 * r - dists
        overlaps = np.where(overlaps > 0, overlaps, 0)
        
        # Normalize diff vector to get direction
        # Avoid division by zero for identical points (unlikely here)
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        dir_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Apply repulsive force proportional to overlap
        # Force magnitude depends on how much they overlap
        for i in range(n):
            for j in range(i + 1, n):
                if overlaps[i, j] > 0:
                    # Push i away from j, j away from i
                    push = overlaps[i, j] * dir_vecs[i, j]
                    forces[i] += push
                    forces[j] -= push

        # Update centers
        centers += forces * 0.05
        
        # Clamp centers to stay roughly inside (soft constraint to prevent exploding)
        np.clip(centers, 1e-6, 1-1e-6, out=centers)

    # 3. Final Safety Check and Output
    # If any constraint is slightly violated due to numerical drift, shrink radius slightly
    # and clamp centers.
    
    # Check boundaries
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    # Shrink r to fit boundaries strictly
    r_boundary = min(min_x, 1 - max_x, min_y, 1 - max_y)
    
    # Check overlaps and shrink r if necessary
    # We can approximate the required shrink by looking at minimum distance
    # between centers.
    # dist >= 2r -> r <= dist/2
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_dist:
                min_dist = d
    
    r_overlap = min_dist / 2
    
    final_r = min(r_boundary, r_overlap)
    
    # Apply final radius
    final_radii = np.full(n, final_r)
    
    # Ensure centers are strictly inside [r, 1-r]
    # If a center is too close to boundary, move it in
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], final_r, 1 - final_r)
        centers[i, 1] = np.clip(centers[i, 1], final_r, 1 - final_r)
        
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii
