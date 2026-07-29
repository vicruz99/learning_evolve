# sol_000281 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8527a4ba) state=53ab83ba sum of radii=1.591661 correctness=1.0
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

def get_hexagonal_grid(n, square_size=1.0):
    """Generate an initial hexagonal grid configuration."""
    # Estimate initial radius to fit n circles roughly
    area_per_circle = square_size**2 / n
    r_est = math.sqrt(area_per_circle / math.pi) * 0.85
    
    rows = int(math.ceil(math.sqrt(n * math.sqrt(3) / 2)))
    centers = []
    radii = []
    idx = 0
    
    # Try to fit circles in a hexagonal lattice
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2 * r
    
    # Adjust r to ensure at least n circles fit in [0,1]x[0,1]
    # Start with a conservative r
    r = 0.08 
    
    while len(centers) < n:
        centers = []
        radii = []
        idx = 0
        row = 0
        y = r
        while y + r <= 1.0 + 1e-6:
            # Determine start x for this row (alternating shift)
            start_x = r if row % 2 == 0 else 2 * r
            x = start_x
            
            # Determine how many circles fit in this row width
            # Width available is 1.0. Each circle takes 2r.
            # But shifted row might need 2r spacing starting from 2r.
            # Actually, standard hex: row 0 at r, 3r, 5r...
            # row 1 at 2r, 4r, 6r...
            
            while x + r <= 1.0 + 1e-6:
                centers.append([x, y])
                radii.append(r)
                idx += 1
                x += 2 * r
                if idx >= n:
                    break
            
            if idx >= n:
                break
            y += r * math.sqrt(3)
            row += 1
        
        if len(centers) < n:
            r *= 0.95 # Decrease radius slightly to fit more
    
    # Pad if necessary (though loop should handle it)
    while len(centers) < n:
        centers.append([0.5, 0.5])
        radii.append(r)
        
    return np.array(centers[:n]), np.array(radii[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize with hexagonal grid
    centers, radii = get_hexagonal_grid(n)
    
    # 2. Force-directed optimization to maximize sum of radii
    # We treat radii as variables that try to grow.
    # We simulate repulsion between circles.
    
    # Initial step size for radii growth
    # Start with small random perturbation to break symmetry if needed
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.001, 0.001, size=centers.shape)
    
    # Parameters
    iterations = 5000
    repulsion_strength = 1.0
    attraction_to_boundary = 0.05 # Pushes circles to touch boundaries
    
    # To maximize sum of radii, we can iteratively:
    # 1. Increase all radii slightly.
    # 2. Resolve collisions by moving centers.
    
    # Better approach: Gradient-like update
    # Maximize sum(r_i)
    # s.t. dist(i,j) >= r_i + r_j
    
    # Let's use a simple iterative relaxation:
    # For each pair, if overlapping, push apart.
    # Also expand radii until constrained.
    
    for step in range(iterations):
        # Decay step size
        alpha = 0.5 / (1 + step * 0.001)
        
        # Try to increase radii
        # A safe way is to compute the max possible radius for each circle
        # given current positions, but this is coupled.
        # Instead, we just grow them all uniformly and then fix overlaps.
        
        growth_rate = 1.0 + 0.001 * (0.995 ** (step / 100.0))
        radii = np.clip(radii * growth_rate, 0, 0.5)
        
        # Resolve overlaps and boundary violations
        for i in range(n):
            # Check boundaries
            for dim in range(2):
                if centers[i, dim] - radii[i] < 0:
                    centers[i, dim] = radii[i]
                elif centers[i, dim] + radii[i] > 1:
                    centers[i, dim] = 1 - radii[i]
            
            # Check other circles
            for j in range(n):
                if i == j: continue
                
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap detected, push apart
                    # Push proportional to overlap and inverse of radius (smaller move more)
                    # But we want to preserve relative sizes.
                    # Simple equal push
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Move i away from j
                    centers[i, 0] += nx * (overlap / 2)
                    centers[i, 1] += ny * (overlap / 2)
                    
                    # Move j away from i (handled when i=j loop or explicitly)
                    # To be symmetric, we can update j's center here or let next iteration handle it.
                    # Explicit update for speed
                    centers[j, 0] -= nx * (overlap / 2)
                    centers[j, 1] -= ny * (overlap / 2)

        # Clamp centers to valid range just in case
        centers = np.clip(centers, 1e-6, 1 - 1e-6)
        
        # Re-adjust radii to fit boundaries strictly
        for i in range(n):
            max_r = min(
                centers[i, 0],
                1 - centers[i, 0],
                centers[i, 1],
                1 - centers[i, 1]
            )
            if radii[i] > max_r:
                radii[i] = max_r

    # Final cleanup: Ensure strict non-overlap and boundary compliance
    # If any overlaps remain, reduce radii
    changed = True
    while changed:
        changed = False
        for i in range(n):
            # Boundary
            r_max = min(
                centers[i, 0],
                1 - centers[i, 0],
                centers[i, 1],
                1 - centers[i, 1]
            )
            if radii[i] > r_max:
                radii[i] = r_max
                changed = True
            
            for j in range(i + 1, n):
                dist = math.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Reduce radii equally to fix
                    overlap = sum_r - dist
                    # Reduce both by half overlap
                    reduction = overlap / 2 + 1e-9
                    radii[i] = max(0, radii[i] - reduction)
                    radii[j] = max(0, radii[j] - reduction)
                    changed = True

    # Validate
    is_valid = validate_packing(centers, radii)
    if not is_valid:
        # Fallback to a valid grid if optimization failed (unlikely)
        centers, radii = get_hexagonal_grid(n)
        # Ensure valid
        for i in range(n):
            r_max = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            radii[i] = min(radii[i], r_max)

    total_sum = np.sum(radii)
    return centers, radii, float(total_sum)

if __name__ == "__main__":
    centers, radii, total_sum = run_packing()
    print(f"Total Sum of Radii: {total_sum}")
    print(f"Validation: {validate_packing(centers, radii)}")
