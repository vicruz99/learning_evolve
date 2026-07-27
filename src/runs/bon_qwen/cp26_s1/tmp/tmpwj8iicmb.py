import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # Number of iterations for the optimization
    iterations = 5000
    
    # Initial radius guess. 
    # For 25 circles, r=0.1 fits in 5x5 grid. 
    # For 26, we expect slightly less if equal, or maybe equal is possible with better layout.
    # Let's start with a reasonable guess.
    current_radius = 0.095
    
    # Initialize centers in a hexagonal-like pattern
    centers = np.zeros((n, 2))
    radii = np.full(n, current_radius)
    
    # Try to fit them in a grid first to generate a good starting point
    # 5x5 grid + 1
    idx = 0
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            if idx < n:
                # Hexagonal shift for odd rows to pack tighter vertically
                # Actually for initialization, just dense packing
                x = 0.1 + j * 0.2 # spacing 0.2
                y = 0.1 + i * 0.2
                centers[idx] = [x, y]
                idx += 1
    
    # If we have more circles than 25 (which we do, 26), place the extra one
    if idx < n:
        # Place remaining in gaps or best available spot
        # Simple random placement in empty space or just append to grid if space
        # For 26, let's try to place in a gap of the 5x5 grid
        # Gaps are at (0.2, 0.2), (0.4, 0.2)...
        # Let's just place it roughly in center of a gap
        # 5x5 grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
        # Gap centers: 0.2, 0.4, 0.6, 0.8
        # Let's place at (0.2, 0.2) but that might be too close.
        # Actually, let's just use a random valid position or force-directed from there.
        # A safe start for the 26th circle:
        centers[idx] = [0.5, 0.5] # Center of square, will be pushed out
        radii[idx] = 0.01 # Start small
        idx += 1
        
    # Ensure all radii are equal initially for symmetry, or keep small for the last one?
    # Better to start all equal to encourage equal packing, which is often optimal or near-optimal.
    # But if we start with r=0.095, the 26th might overlap heavily.
    # Let's reduce initial radius to ensure valid start.
    start_r = 0.05
    radii[:] = start_r
    
    # Optimization parameters
    learning_rate = 1e-4
    repulsion_strength = 1.0
    expansion_rate = 1e-6
    
    for step in range(iterations):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # 1. Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap amount
                    overlap = min_dist - dist
                    # Force proportional to overlap, inversely proportional to distance to avoid singularity
                    force_magnitude = repulsion_strength * overlap / (dist + 1e-9)
                    direction = diff / dist
                    forces[i] += direction * force_magnitude
                    forces[j] -= direction * force_magnitude
                elif dist == 0:
                    # Random nudge if centers coincide
                    forces[i] += np.random.rand(2) * 0.01
                    forces[j] -= np.random.rand(2) * 0.01

        # 2. Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += repulsion_strength * (r - x)
            # Right wall
            if x + r > 1:
                forces[i, 0] -= repulsion_strength * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += repulsion_strength * (r - y)
            # Top wall
            if y + r > 1:
                forces[i, 1] -= repulsion_strength * (y + r - 1)
                
        # Update positions
        # Adaptive step size: if step is large, move more aggressively?
        # Simple fixed step for stability, decaying over time
        current_lr = learning_rate * (0.999 ** step)
        centers += current_lr * forces
        
        # Clamp centers to valid range [r, 1-r] to prevent divergence
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
            
        # Gradually expand radii
        # We want to maximize sum of radii. 
        # If system is stable (low forces), expand.
        # A simple strategy: increase radius slightly every step, 
        # but if collisions occur frequently, maybe hold or decrease?
        # Given the repulsion forces, we can just increase radius slowly.
        radii += expansion_rate * (0.9995 ** step)
        
        # Cap radius to prevent exploding if something goes wrong
        radii = np.clip(radii, 0.01, 0.5)
        
        # Check for validity (optional, for debugging)
        # if step % 1000 == 0:
        #     print(f"Step {step}, Avg R: {np.mean(radii):.4f}")

    # Final refinement: Scale down slightly if any violations remain due to numerical error
    # Check max overlap
    max_violation = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            req = radii[i] + radii[j]
            if dist < req:
                max_violation = max(max_violation, req - dist)
        
        # Boundary
        r = radii[i]
        x, y = centers[i]
        if x - r < 0: max_violation = max(max_violation, r - x)
        if x + r > 1: max_violation = max(max_violation, x + r - 1)
        if y - r < 0: max_violation = max(max_violation, r - y)
        if y + r > 1: max_violation = max(max_violation, y + r - 1)

    # If there are violations, shrink radii uniformly to fix them
    if max_violation > 1e-9:
        # Estimate needed reduction. 
        # Shrinking radii by delta reduces required dist by 2*delta.
        # So reduce radii by max_violation / 2
        shrink = max_violation / 2 + 1e-6
        radii -= shrink
        radii = np.clip(radii, 0, None)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To verify locally
if __name__ == "__main__":
    import numpy as np # Ensure numpy is available
    
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any():
            return False
        for i in range(n):
            if radii[i] < 0: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")