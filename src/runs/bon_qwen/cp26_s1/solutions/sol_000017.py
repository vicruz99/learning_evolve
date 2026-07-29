# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=9cd7f179 sum of radii=0.508854 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42) # For reproducibility

    # 1. Initialization: Hexagonal Grid
    # We arrange points in a hexagonal lattice.
    # For n=26, a 5x5 grid is 25 points. We can add one point or use a distorted grid.
    # Let's try to place them in rows. 
    # Rows: 5, 6, 5, 6, 4 (Total 26) or similar. 
    # Actually, a simple rectangular grid is easier to perturb later.
    # Let's use a 5x5 grid plus one extra point, or just 26 points in a grid.
    
    centers = np.zeros((n, 2))
    
    # Simple grid initialization for robustness, then we optimize
    # Let's try to distribute them roughly evenly.
    # 26 points. Sqrt(26) ~ 5.1. 
    # Let's do 6 columns, 5 rows (30 slots) and pick 26?
    # Or just linear indexing on a grid.
    
    # Better: Hexagonal packing logic.
    # Rows of circles. 
    # Row 0: 5 circles
    # Row 1: 6 circles (offset)
    # Row 2: 5 circles
    # Row 3: 6 circles
    # Row 4: 4 circles
    # Total: 26.
    
    rows_config = [5, 6, 5, 6, 4]
    
    # Estimate radius to fit this in unit square
    # Height: 5 rows. Vertical spacing sqrt(3)/2 * diameter.
    # Width: max row width.
    # Let's just place them randomly first? No, grid is better.
    # Let's place them in a 5x6 grid (30) and remove 4?
    # Or just use a uniform grid.
    
    # Let's try a uniform grid of 6x5, take first 26.
    x_coords = np.linspace(0.1, 0.9, 6) # 6 cols
    y_coords = np.linspace(0.1, 0.9, 5) # 5 rows
    # This gives 30 points.
    
    idx = 0
    temp_centers = []
    for y in y_coords:
        for x in x_coords:
            if idx < n:
                temp_centers.append([x, y])
                idx += 1
            else:
                break
        if idx >= n:
            break
    
    centers = np.array(temp_centers)
    
    # Initialize radii
    # Start small to ensure validity
    radii = np.full(n, 0.02)

    # 2. Optimization Loop
    # We want to maximize sum(radii).
    # We will use an iterative approach:
    # 1. Try to increase radii.
    # 2. If invalid, move centers to resolve conflicts.
    
    # Number of iterations
    iterations = 2000
    
    # Helper to check constraints and compute overlaps
    def get_conflicts(c, r):
        # Returns a list of (i, j, overlap) and boundary violations
        conflicts = []
        n_c = len(c)
        for i in range(n_c):
            # Boundary checks
            if c[i, 0] - r[i] < 0:
                conflicts.append(('left', i, r[i] - c[i, 0]))
            if c[i, 0] + r[i] > 1:
                conflicts.append(('right', i, c[i, 0] + r[i] - 1))
            if c[i, 1] - r[i] < 0:
                conflicts.append(('bottom', i, r[i] - c[i, 1]))
            if c[i, 1] + r[i] > 1:
                conflicts.append(('top', i, c[i, 1] + r[i] - 1))
            
            for j in range(i + 1, n_c):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                overlap = (r[i] + r[j]) - dist
                if overlap > 1e-12:
                    conflicts.append(('pair', i, j, overlap))
        return conflicts

    # We will use a "Repulsion" simulation
    # Forces:
    # - Between circles: Repulsion if dist < r_i + r_j + margin
    # - From boundaries: Repulsion if dist to boundary < r_i + margin
    
    for t in range(iterations):
        # Cooling schedule for step size and repulsion strength
        temp = 1.0 - (t / iterations)
        step_size = 0.001 + 0.01 * temp
        
        # Forces array
        forces = np.zeros_like(centers)
        
        # Calculate pairwise repulsion
        # We only care if they are close or overlapping
        # To maximize radii, we want to push them apart as much as possible
        # But we also need to allow radii to grow.
        # Strategy: Fix radii for a moment, push centers apart.
        
        # Calculate distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
        dists = np.sqrt(np.sum(diffs**2, axis=2)) # (n, n)
        np.fill_diagonal(dists, np.inf) # Ignore self
        
        # Current radii sum
        r_sum = np.sum(radii)
        
        # If radii are too small, try to inflate them uniformly?
        # Or just rely on the fact that if we push centers apart, we can increase radii later.
        
        # Let's try to increase radii slightly every few steps if valid
        if t % 5 == 0:
            # Try to increase radii
            # Check current validity with current radii
            valid = True
            # Quick check
            # We can compute the max possible radius for each circle given current positions
            # r_i <= min(boundaries, min_j(d_ij - r_j))
            # This is hard to solve analytically fast.
            # Let's just try to scale all radii up by a small factor
            scale_factor = 1.001
            new_radii = radii * scale_factor
            
            # Check if valid
            is_valid = True
            # Boundary check
            if np.any(centers[:, 0] - new_radii < -1e-9) or \
               np.any(centers[:, 0] + new_radii > 1 + 1e-9) or \
               np.any(centers[:, 1] - new_radii < -1e-9) or \
               np.any(centers[:, 1] + new_radii > 1 + 1e-9):
                is_valid = False
            
            if is_valid:
                # Pairwise check
                # This is O(N^2) but N=26 is small
                for i in range(n):
                    for j in range(i+1, n):
                        if dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid:
                radii = new_radii
        
        # Calculate forces to resolve overlaps and optimize spacing
        # We treat the target distance as r_i + r_j.
        # If dist < r_i + r_j, push apart.
        # Even if dist > r_i + r_j, we might want to push apart to create room for larger radii?
        # Yes, a repulsive force like 1/d^2 helps spread points.
        
        # Repulsion force from pairs
        # Force magnitude = k / dist^2 (if dist is small) or 0?
        # To maximize min distance, we want to push everything apart.
        
        # Let's use a force that is active when dist < r_i + r_j + buffer
        buffer = 0.05 # Allow some slack
        
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < buffer + radii[i] + radii[j] + 0.05: # Active region
                    if d < 1e-6:
                        d = 1e-6
                    # Repulsion force
                    # Direction: i away from j
                    dir_vec = (centers[i] - centers[j]) / d
                    # Force magnitude increases as distance decreases
                    force_mag = 0.1 / (d * d + 1e-6) 
                    # If overlapping, stronger force
                    if d < radii[i] + radii[j]:
                        force_mag *= 10.0
                    
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag

        # Boundary forces
        # Push centers away from boundaries
        # If x < r, push right. If x > 1-r, push left.
        # Actually, to maximize r, we want to be far from boundaries.
        # Force towards center if close to boundary.
        
        # Check x bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left
            if x < r + 0.05:
                forces[i, 0] += 0.5 * (r + 0.05 - x)
            # Right
            if x > 1 - r - 0.05:
                forces[i, 0] -= 0.5 * (x - (1 - r - 0.05))
            # Bottom
            if y < r + 0.05:
                forces[i, 1] += 0.5 * (r + 0.05 - y)
            # Top
            if y > 1 - r - 0.05:
                forces[i, 1] -= 0.5 * (y - (1 - r - 0.05))

        # Apply forces
        # Limit step size to avoid instability
        max_disp = np.max(np.abs(forces))
        if max_disp > 1e-6:
            centers += forces * step_size
        
        # Project to valid range [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Random jitter to escape local minima
        if t % 50 == 0:
            jitter = np.random.normal(0, 0.01, size=centers.shape)
            centers += jitter
            centers = np.clip(centers, 0, 1)

    # Final refinement:
    # With fixed centers, find the largest possible radii.
    # This is solving a linear problem? No.
    # But we can just shrink radii until valid.
    
    # Iterative shrink
    for _ in range(100):
        valid = True
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < radii[i] + radii[j] - 1e-12:
                    # Reduce radii to just touch
                    # Ideally distribute the reduction, but let's just halve the excess
                    # Or set sum to d
                    sum_r = radii[i] + radii[j]
                    ratio = d / sum_r
                    radii[i] *= ratio
                    radii[j] *= ratio
                    valid = False
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            min_dist = min(x, 1-x, y, 1-y)
            if r > min_dist + 1e-12:
                radii[i] = min_dist
                valid = False
        
        if valid:
            break

    # Final validation and correction
    # Ensure strict validity
    for _ in range(10):
        changed = False
        for i in range(n):
            # Boundary
            x, y = centers[i]
            r = radii[i]
            max_r_boundary = min(x, 1-x, y, 1-y)
            if r > max_r_boundary + 1e-12:
                radii[i] = max_r_boundary
                changed = True
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if d < sum_r - 1e-12:
                    # Reduce proportionally
                    factor = d / sum_r
                    radii[i] *= factor
                    radii[j] *= factor
                    changed = True
        if not changed:
            break

    sum_radii = np.sum(radii)
    
    # Final sanity check
    if not validate_packing(centers, radii):
        print("Validation failed, attempting recovery...")
        # Fallback to smaller radii
        radii *= 0.9
        while not validate_packing(centers, radii):
            radii *= 0.99
        sum_radii = np.sum(radii)

    return centers, radii, float(sum_radii)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import numpy as np
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

# To run and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
