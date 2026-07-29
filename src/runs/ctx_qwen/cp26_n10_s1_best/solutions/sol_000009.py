# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d25b46ef) state=9579ee4a sum of radii=2.424498 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Perturbed 5x5 grid + 1 central circle
    # A 5x5 grid allows r=0.1. We start there and optimize.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    idx = 0
    # Create 5x5 grid points in (0,1)
    grid_size = 5
    spacing = 1.0 / (grid_size + 1) # Start slightly inside to allow expansion
    
    # We want centers roughly at 0.1, 0.3, 0.5, 0.7, 0.9
    # But let's place them at 0.1 + k*0.2? 
    # If r=0.1, center must be >= 0.1.
    # Let's place centers at 0.1, 0.3, 0.5, 0.7, 0.9
    coords = np.linspace(0.1, 0.9, 5)
    
    for i in range(5):
        for j in range(5):
            if idx < n:
                centers[idx] = [coords[i] + np.random.normal(0, 0.005), 
                                coords[j] + np.random.normal(0, 0.005)]
                radii[idx] = 0.09 # Start slightly smaller to ensure validity before expansion
                idx += 1
                
    # 26th circle in the center gap
    if idx < n:
        centers[idx] = [0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)]
        radii[idx] = 0.04 # Small radius for the gap filler
        idx += 1
        
    # Clamp initial positions to valid range
    for i in range(n):
        centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])

    # 2. Optimization Loop
    # We will iteratively try to expand radii and resolve collisions
    num_iterations = 2000
    dt = 0.0005
    
    for iter_num in range(num_iterations):
        # Try to increase radii
        # A simple heuristic: increase all radii slightly
        # Or increase based on "free space"
        
        # Let's try to grow each circle by a small amount
        growth_rate = 1e-4 * (1 - iter_num/num_iterations) # Annealing growth
        # But we can't just grow, we must push others away
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            
            # Boundary forces (push away from walls)
            # If too close to wall, push in
            # Actually, we want to maximize radius, so we want to be centered in available space.
            # But boundary constraints are hard.
            # If x - r < 0, we must move x to the right.
            
            # Collision resolution and expansion
            for j in range(i + 1, n):
                xj, yj = centers[j]
                rj = radii[j]
                
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-9:
                    dist = 1e-9
                    dx, dy = 1e-9, 0.0
                
                min_dist = ri + rj
                
                if dist < min_dist:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    # Push proportional to radius? Or equally?
                    # To maximize sum of radii, we want to keep them large.
                    # Separating them creates space for growth.
                    
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Move i away from j, j away from i
                    # Move distance proportional to overlap
                    # Let's move them apart by half the overlap each
                    move_i = -0.5 * overlap
                    move_j = 0.5 * overlap
                    
                    centers[i, 0] += nx * move_i
                    centers[i, 1] += ny * move_i
                    centers[j, 0] += nx * move_j
                    centers[j, 1] += ny * move_j
                    
                    # Also, if they are stuck, we might need to reduce radii slightly?
                    # But we want to maximize sum.
                    # If we push them apart, we might create space to increase radii later.
        
        # Boundary constraints enforcement
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            
            # Check boundaries
            # x must be >= r and <= 1-r
            # y must be >= r and <= 1-r
            
            # If violation, move center to satisfy
            if xi < ri:
                centers[i, 0] = ri
            elif xi > 1 - ri:
                centers[i, 0] = 1 - ri
                
            if yi < ri:
                centers[i, 1] = ri
            elif yi > 1 - ri:
                centers[i, 1] = 1 - ri
                
            # If after moving, it's still invalid (e.g. r > 0.5), reduce r
            # But r shouldn't be that large.
            if ri > 0.5:
                radii[i] = 0.5
                centers[i] = [0.5, 0.5]

        # Expansion Step
        # Try to increase radii
        # A safe way is to find the minimum distance to any neighbor or wall,
        # and set radius to half of that? No, that's for equal circles.
        
        # Heuristic: increase radius by a small amount, then resolve overlaps.
        # This is effectively what the loop does if we increase radii here.
        
        # Let's increase radii slightly
        # Avoid increasing if already large or if stuck
        for i in range(n):
            # Check min distance to others and walls
            min_d = 1.0 # Max possible
            
            # Walls
            min_d = min(min_d, centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            # Neighbors
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < 1e-9: dist = 1e-9
                # Space available is dist - r_j
                space = dist - radii[j]
                if space < min_d:
                    min_d = space
            
            # Max possible radius for circle i is min_d
            # But we want to grow.
            # If current r < min_d, we can grow.
            # But growing might cause overlap if others grow too.
            # However, if we push others away in the collision step, we create space.
            
            # Let's just grow slightly
            if radii[i] < min_d - 1e-9:
                radii[i] += growth_rate
                # Cap at max possible
                if radii[i] > min_d:
                    radii[i] = min_d
                    
    # 3. Final Validation and Cleaning
    # Ensure strict validity
    for _ in range(100): # Quick cleanup iterations
        changed = False
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            
            # Boundary check
            if xi - ri < 0:
                centers[i, 0] = ri
                changed = True
            if xi + ri > 1:
                centers[i, 0] = 1 - ri
                changed = True
            if yi - ri < 0:
                centers[i, 1] = ri
                changed = True
            if yi + ri > 1:
                centers[i, 1] = 1 - ri
                changed = True
            
            # Overlap check - push out
            for j in range(i + 1, n):
                xj, yj = centers[j]
                rj = radii[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                req = ri + rj
                
                if dist < req - 1e-12 and dist > 1e-9:
                    overlap = req - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Move apart
                    centers[i, 0] -= nx * overlap * 0.5
                    centers[i, 1] -= ny * overlap * 0.5
                    centers[j, 0] += nx * overlap * 0.5
                    centers[j, 1] += ny * overlap * 0.5
                    changed = True
                elif dist < 1e-9:
                    # Push randomly
                    centers[i, 0] -= 0.001
                    centers[j, 0] += 0.001
                    changed = True
        
        if not changed:
            break

    # Recalculate sum
    sum_radii = np.sum(radii)
    
    # Final validation
    is_valid = validate_packing(centers, radii)
    if not is_valid:
        print("Warning: Packing validation failed. Attempting to fix...")
        # Fallback to a known valid packing if optimization fails
        # Simple grid
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.09) # Safe radius
        idx = 0
        for r in range(5):
            for c in range(5):
                centers[idx] = [0.1 + r*0.2, 0.1 + c*0.2]
                idx += 1
        centers[25] = [0.5, 0.5]
        radii[25] = 0.05
        # Re-validate
        is_valid = validate_packing(centers, radii)
        sum_radii = np.sum(radii)

    return centers, radii, float(sum_radii)

# Test run
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
    print(f"Radii: {radii}")
    print(f"Centers: {centers}")
