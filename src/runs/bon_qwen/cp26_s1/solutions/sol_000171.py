# sol_000171 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8150d860) state=abc9572e sum of radii=0.194670 correctness=1.0
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
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        centers: np.array (26, 2)
        radii: np.array (26,)
        sum_radii: float
    """
    n_circles = 26
    np.random.seed(42) # For reproducibility

    # --- Initialization: Hexagonal Grid ---
    # We try to fit 26 circles in a hexagonal pattern.
    # A 5x5 grid is 25 circles. We add 1 more.
    # Hexagonal packing is denser.
    # Let's try rows of 6, 5, 6, 5, 4 (Total 26) or 5, 6, 5, 6, 4?
    # Actually, let's just generate a dense grid and trim.
    
    # Generate points in a hexagonal lattice
    # Spacing factor will be adjusted during optimization, but let's start reasonable.
    # Assume r ~ 0.1, so spacing ~ 0.2.
    
    centers = []
    r_init = 0.05 # Start small
    spacing = 2 * r_init * 1.5 # Rough guess
    
    # Generate a dense hex grid that covers the square
    rows = 15
    cols = 15
    for r_idx in range(rows):
        for c_idx in range(cols):
            x = c_idx * spacing * 0.866 # cos(30)
            y = r_idx * spacing * 0.5   # sin(30)
            if r_idx % 2 == 1:
                x += spacing * 0.866 / 2
            
            # Add point if inside square (with some margin)
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
    
    centers = np.array(centers)
    
    # If we have more than 26, pick the first 26 (or best 26? First 26 is fine for init)
    # Actually, let's pick the first 26 from the list.
    if len(centers) > n_circles:
        centers = centers[:n_circles]
    
    # Ensure shape
    centers = centers[:n_circles]
    
    # Current radius for optimization
    current_r = 0.01
    max_r = 0.01
    
    # Optimization parameters
    steps_per_iteration = 200
    learning_rate = 0.005
    r_increment = 0.0005
    max_r_target = 0.15 # Safety cap
    
    # Main Optimization Loop: Expand and Relax
    # We iteratively increase radius and resolve collisions
    for iteration in range(1000): # Max iterations
        current_r += r_increment
        
        # If we exceeded a safe bound or progress stalled, break (though logic handles it)
        if current_r > max_r_target:
            current_r -= r_increment
            break
            
        # Relaxation loop for current radius
        resolved = False
        for step in range(steps_per_iteration):
            forces = np.zeros_like(centers)
            overlap_energy = 0.0
            
            # 1. Boundary Forces
            # Push centers away from boundaries if r > dist to boundary
            for i in range(n_circles):
                x, y = centers[i]
                
                # Left wall
                if x - current_r < 0:
                    overlap = current_r - x
                    forces[i, 0] += overlap # Push right
                    overlap_energy += overlap**2
                # Right wall
                if x + current_r > 1:
                    overlap = (x + current_r) - 1
                    forces[i, 0] -= overlap # Push left
                    overlap_energy += overlap**2
                # Bottom wall
                if y - current_r < 0:
                    overlap = current_r - y
                    forces[i, 1] += overlap # Push up
                    overlap_energy += overlap**2
                # Top wall
                if y + current_r > 1:
                    overlap = (y + current_r) - 1
                    forces[i, 1] -= overlap # Push down
                    overlap_energy += overlap**2

            # 2. Pairwise Repulsion Forces
            # Vectorized computation for speed
            # diff[i, j] = centers[i] - centers[j]
            # dist[i, j] = norm(diff)
            # if dist < 2*r, force = (2*r - dist) * direction
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
            dist_sq = np.sum(diff**2, axis=2) # (n, n)
            dist_sq = np.maximum(dist_sq, 1e-12) # Avoid div by zero
            dist = np.sqrt(dist_sq)
            
            # Mask for overlapping pairs (strictly less than 2r)
            # We want to push apart if dist < 2*current_r
            # Force magnitude proportional to overlap
            overlap_dist = 2 * current_r - dist
            is_colliding = (overlap_dist > 1e-6)
            
            # Direction vector (normalized)
            # Avoid division by zero
            safe_dist = np.where(dist < 1e-9, 1e-9, dist)
            directions = diff / safe_dist[:, :, np.newaxis] # (n, n, 2)
            
            # Apply forces: F_ij = overlap * direction_ij
            # F_i += sum_j (overlap * direction)
            # Note: direction from i to j is (center[j] - center[i])? 
            # diff[i, j] = center[i] - center[j]. So direction is i->j? No.
            # If center[i] is close to center[j], we want to push i away from j.
            # Vector from j to i is center[i] - center[j] = diff[i, j].
            # So we add force in direction of diff[i, j].
            
            force_magnitudes = np.where(is_colliding, overlap_dist, 0.0)
            forces += np.sum(force_magnitudes[:, :, np.newaxis] * directions, axis=1)
            
            overlap_energy += np.sum(force_magnitudes**2)
            
            # Update positions
            # Damping to stabilize
            centers += learning_rate * forces
            centers = np.clip(centers, 0, 1) # Keep inside bounds roughly (forces handle it better but clip helps)
            
            # If energy is very low, we are stable for this radius
            if overlap_energy < 1e-8:
                resolved = True
                break
        
        if resolved:
            max_r = current_r
        else:
            # If we couldn't resolve collisions after many steps, back off radius slightly
            # or stop. Here we just stop increasing r effectively if stuck.
            # But we want to keep trying to find a better config.
            # A simple strategy: reduce learning rate or accept smaller r.
            # For this simple solver, if not resolved, we revert r increment?
            # Let's just keep current max_r and break to save time if stuck deep.
            if iteration > 100:
                 # Revert radius increase
                 current_r -= r_increment
                 # Optional: perturb centers to escape local minima
                 centers += np.random.normal(0, 0.01, centers.shape)
                 centers = np.clip(centers, 0, 1)
                 current_r -= r_increment # Back up
                 max_r = current_r
                 # Try smaller increment
                 r_increment *= 0.5
                 if r_increment < 1e-6:
                     break
            else:
                 current_r -= r_increment
                 max_r = current_r

    # --- Final Radius Calculation ---
    # Now that centers are fixed, calculate the maximum valid radius for each circle.
    # This allows for unequal radii if gaps permit, though they will be close to max_r.
    radii = np.full(n_circles, 1.0) # Start large

    # Constraint from boundaries
    x = centers[:, 0]
    y = centers[:, 1]
    radii = np.minimum(radii, np.minimum(x, 1 - x))
    radii = np.minimum(radii, np.minimum(y, 1 - y))

    # Constraint from neighbors
    # For each circle i, r_i <= dist(i, j) - r_j
    # This is a system of inequalities. We can solve it iteratively or just use min(dist/2).
    # Using min(dist/2) assumes equal radii, which is a safe lower bound.
    # To get the true max sum, we could solve the LP, but min(dist/2) is a good approximation
    # and ensures validity.
    # However, to strictly maximize sum, we should solve for radii.
    # But given the tight packing, r_i ~ r_j.
    # Let's use the relaxation: r_i = min( min_j (dist_ij - r_j), boundary )
    # We can iterate this.
    
    current_radii = np.full(n_circles, max_r)
    for _ in range(20): # Iterate to converge radii
        next_radii = np.copy(current_radii)
        # Boundary constraints
        next_radii = np.minimum(next_radii, np.minimum(x, 1 - x))
        next_radii = np.minimum(next_radii, np.minimum(y, 1 - y))
        
        # Neighbor constraints
        # r_i + r_j <= dist_ij  => r_i <= dist_ij - r_j
        for i in range(n_circles):
            for j in range(n_circles):
                if i != j:
                    d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    limit = d - current_radii[j]
                    if limit < next_radii[i]:
                        next_radii[i] = limit
        
        # Ensure non-negative
        next_radii = np.maximum(next_radii, 0.0)
        current_radii = next_radii
        
        # Check convergence
        if np.max(np.abs(current_radii - next_radii)) < 1e-7:
            break
            
    radii = current_radii
    sum_radii = np.sum(radii)

    # Validate
    is_valid = validate_packing(centers, radii)
    if not is_valid:
        # Fallback to equal radii if validation fails (shouldn't happen with logic above)
        # But just in case, clamp radii
        radii = np.full(n_circles, 0.0) # Should not reach here with correct logic
        # Re-calculate safe equal radius
        min_dist = np.inf
        for i in range(n_circles):
            # Boundary
            d_b = min(x[i], 1-x[i], y[i], 1-y[i])
            min_dist = min(min_dist, d_b)
            for j in range(i+1, n_circles):
                d_ij = np.sqrt(np.sum((centers[i]-centers[j])**2))
                min_dist = min(min_dist, d_ij / 2)
        
        radii = np.full(n_circles, min_dist)
        sum_radii = np.sum(radii)
        # Re-validate
        validate_packing(centers, radii)

    return centers, radii, sum_radii

# For testing purposes if run directly
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radii: {radii}")
    print(f"Valid: {validate_packing(centers, radii)}")
