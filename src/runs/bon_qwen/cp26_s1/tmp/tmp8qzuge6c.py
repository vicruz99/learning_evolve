import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a heuristic initialization followed by iterative relaxation.
    """
    n = 26
    
    # --- Step 1: Initialization ---
    # We use a staggered (hexagonal-like) grid. 
    # 5 rows with varying counts to reach 26 circles (6, 5, 6, 5, 4).
    # This provides a good starting topology for local optimization.
    centers = []
    row_counts = [6, 5, 6, 5, 4]
    
    # Vertical spacing (approximate for r ~ 0.1)
    y_spacing = 0.18 
    y_start = 0.12
    
    for i, count in enumerate(row_counts):
        y = y_start + i * y_spacing
        # Calculate x coordinates to fit 'count' circles roughly in the unit square
        # We leave some margin initially to avoid boundary conflicts
        margin = 0.1
        # Distribute 'count' circles in [margin, 1-margin]
        if count == 1:
            x_pos = 0.5
        else:
            # For hexagonal packing, we might shift odd/even rows
            # But for initial placement, a linear spread is fine
            spacing = (1 - 2 * margin) / (count - 1)
            xs = [margin + j * spacing for j in range(count)]
            # Apply slight horizontal offset for odd rows to mimic hexagonal
            if i % 2 == 1:
                xs = [x + spacing / 2 for x in xs]
                # Clamp to ensure we stay roughly within bounds
                xs = [max(margin, min(1-margin, x)) for x in xs]
            x_pos = xs
        
        for x in x_pos:
            centers.append([x, y])

    centers = np.array(centers)
    # Start with a reasonable radius (slightly smaller than target to allow growth)
    radii = np.full(n, 0.08)

    # --- Step 2: Iterative Optimization ---
    # We will grow radii and resolve overlaps/boundaries using a force-directed method
    
    steps = 3000 # Number of optimization steps
    growth_rate = 0.0001 # How much to increase radius per step
    repulsion_strength = 0.5 # Force magnitude for overlap resolution
    boundary_strength = 1.0 # Force magnitude to keep circles inside

    for step in range(steps):
        # 1. Increase radii
        radii += growth_rate
        
        # 2. Calculate velocities based on forces
        velocities = np.zeros_like(centers)
        
        # Apply boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                velocities[i, 0] += boundary_strength * (r - x)
            # Right wall
            if x + r > 1:
                velocities[i, 0] -= boundary_strength * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                velocities[i, 1] += boundary_strength * (r - y)
            # Top wall
            if y + r > 1:
                velocities[i, 1] -= boundary_strength * (y + r - 1)
            
            # Soft centering force to keep circles away from walls if not touching
            # This helps distribute them better
            if x - r < 0.05: velocities[i, 0] += 0.1
            if x + r > 0.95: velocities[i, 0] -= 0.1
            if y - r < 0.05: velocities[i, 1] += 0.1
            if y + r > 0.95: velocities[i, 1] -= 0.1

        # Apply repulsion forces for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap amount
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    # Normalize direction vector
                    fx = dx / dist
                    fy = dy / dist
                    
                    force = repulsion_strength * overlap
                    
                    velocities[j, 0] += fx * force
                    velocities[j, 1] += fy * force
                    velocities[i, 0] -= fx * force
                    velocities[i, 1] -= fy * force
                elif dist < 1e-9:
                    # If on top of each other, push randomly
                    velocities[i, 0] -= 0.01
                    velocities[i, 1] -= 0.01
                    velocities[j, 0] += 0.01
                    velocities[j, 1] += 0.01

        # Update centers
        centers += velocities
        
        # Clamp centers to unit square [0, 1] x [0, 1] to prevent drifting out
        centers[:, 0] = np.clip(centers[:, 0], 0, 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, 1)

    # --- Step 3: Final Cleanup ---
    # Ensure strict validity and return
    # Check if any circles are still significantly overlapping or out of bounds
    # The iterative process usually handles this, but we can do a final clamp/adjust
    
    # Final radius check: shrink slightly if any overlaps remain to ensure validation passes
    # But since we optimized for growth, we should be at the limit.
    # Let's ensure bounds are respected strictly
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # If center is too close to boundary relative to radius, shrink radius
        r = min(r, x, 1-x, y, 1-y)
        radii[i] = r
        centers[i, 0] = max(r, min(1-r, centers[i, 0]))
        centers[i, 1] = max(r, min(1-r, centers[i, 1]))

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    # Simple self-test
    import numpy as np
    
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

    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")