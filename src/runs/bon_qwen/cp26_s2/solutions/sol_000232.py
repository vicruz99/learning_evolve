# sol_000232 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=da0eccf0 sum of radii=1.634217 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We generate a hexagonal grid and select 26 points that fit well in the square.
    # We start with a relatively small radius to ensure validity, then optimize.
    
    # Estimate a starting radius. 
    # For 26 circles, area ~ 0.83 -> r ~ 0.1. 
    # Start slightly smaller to be safe.
    r_init = 0.05
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    idx = 0
    y = r_init
    row = 0
    # Hexagonal spacing
    dx = 2 * r_init
    dy = math.sqrt(3) * r_init
    
    while idx < n:
        x = r_init
        if row % 2 == 1:
            x += r_init  # Shift for hexagonal pattern
        
        while x + r_init <= 1.0 and idx < n:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
            x += dx
        
        y += dy
        row += 1
        if y + r_init > 1.0 and idx < n:
            # If we run out of vertical space but still need circles,
            # we can try to squeeze them or just break (though with r=0.05 this shouldn't happen)
            # For robustness, if we can't place, we might need a fallback, 
            # but 26 circles of r=0.05 is very sparse, so it will fit easily.
            pass

    # 2. Optimization Loop
    # We will run a number of iterations to grow the circles and adjust positions.
    num_iterations = 1000
    learning_rate = 0.05 # Step size for moving centers
    
    for it in range(num_iterations):
        # Calculate current max possible radii based on geometry
        # But we want to maximize sum, so we try to increase radii.
        # However, simply increasing radii might cause overlaps.
        # Strategy: 
        # a) Calculate "tightness" or constraints.
        # b) Move centers to relieve constraints (repulsion).
        # c) Increase radii.
        
        # Let's compute the "blocking" distances.
        # For each pair, dist_ij. We need dist_ij >= r_i + r_j.
        # If dist_ij < r_i + r_j, we have overlap (should not happen if we control growth).
        # But to grow, we need space.
        
        # Simple approach:
        # 1. Try to increase all radii by a small factor.
        # 2. If overlap occurs, resolve by moving centers apart.
        
        growth_factor = 1.0 + 1.0 / (it + 10) # Slow down growth over time
        radii *= growth_factor
        
        # Resolve overlaps by pushing centers apart
        # We can do a few sub-steps of relaxation
        num_relax_steps = 20
        for _ in range(num_relax_steps):
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    c_i = centers[i]
                    c_j = centers[j]
                    diff = c_j - c_i
                    dist = np.linalg.norm(diff)
                    
                    r_sum = radii[i] + radii[j]
                    
                    if dist < r_sum:
                        # Overlap! Push apart.
                        # Force magnitude proportional to overlap amount
                        overlap = r_sum - dist
                        if dist > 1e-6:
                            force_mag = overlap * 1.0 # Stiff spring
                            direction = diff / dist
                            forces[i] -= force_mag * direction
                            forces[j] += force_mag * direction
                    elif dist < r_sum * 1.1:
                        # Proximity repulsion to prevent getting stuck
                        # This helps optimize the layout
                        force_mag = 0.1 * (1.1 * r_sum - dist)
                        if dist > 1e-6:
                            direction = diff / dist
                            forces[i] -= force_mag * direction
                            forces[j] += force_mag * direction
            
            # Wall repulsion
            for i in range(n):
                r = radii[i]
                x, y = centers[i]
                
                # Left wall
                if x - r < 0:
                    forces[i, 0] += (r - x) * 1.0 # Push right
                elif x - r < 0.05:
                    forces[i, 0] += 0.1 * (0.05 - (x - r))
                    
                # Right wall
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1) * 1.0 # Push left
                elif x + r > 0.95:
                    forces[i, 0] -= 0.1 * ((x + r) - 0.95)
                    
                # Bottom wall
                if y - r < 0:
                    forces[i, 1] += (r - y) * 1.0
                elif y - r < 0.05:
                    forces[i, 1] += 0.1 * (0.05 - (y - r))
                    
                # Top wall
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1) * 1.0
                elif y + r > 0.95:
                    forces[i, 1] -= 0.1 * ((y + r) - 0.95)

            # Apply forces
            centers += learning_rate * forces
            
            # Clamp centers to valid range [r, 1-r] approximately to prevent wild moves
            # But we must respect r. 
            # Better: just ensure they stay in [0, 1]
            centers = np.clip(centers, 0, 1)
            
            # Strictly enforce boundary constraints on radii after move
            # If a center is at x, r can be at most min(x, 1-x)
            # But we are growing r. If we moved center, we might need to reduce r?
            # No, the goal is to find a config where r is large.
            # If we move center to a spot where r doesn't fit, we are in trouble.
            # The repulsion from walls handles this.
            
            # However, to be safe, let's clamp radii to valid max given centers
            for i in range(n):
                max_r_boundary = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
                if radii[i] > max_r_boundary + 1e-9:
                    radii[i] = max_r_boundary

        # Post-iteration: Trim radii to strictly valid values based on current centers
        # This ensures we don't claim a radius that overlaps.
        # We calculate the exact max radius allowed for each circle given others.
        # But doing this sequentially is hard (circular dependency).
        # Instead, we just ensure no overlaps by reducing radii if necessary.
        
        # Check overlaps and reduce larger radius? 
        # Or just clamp?
        # Let's do a passive check: if overlap, reduce radii equally.
        overlap_found = True
        max_pass = 5
        p = 0
        while overlap_found and p < max_pass:
            overlap_found = False
            p += 1
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    req_dist = radii[i] + radii[j]
                    if dist < req_dist - 1e-9:
                        overlap_found = True
                        # Reduce radii
                        reduction = (req_dist - dist) / 2 + 1e-5
                        radii[i] -= reduction
                        radii[j] -= reduction

        # Clamp radii to 0
        radii = np.maximum(radii, 0)

    # Final polish: Try to increase radii one last time until hitting constraints
    # We can't easily do this without moving centers, but centers are optimized.
    # Just ensure validity.
    
    # Check if any radius is invalid due to boundaries
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    # Final overlap check and repair
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - 1e-9:
                # Reduce the one with larger radius or both
                r_sum = radii[i] + radii[j]
                diff = r_sum - dist
                # Reduce proportionally
                total_r = radii[i] + radii[j]
                if total_r > 1e-12:
                    factor_i = radii[i] / total_r
                    factor_j = radii[j] / total_r
                    radii[i] -= diff * factor_i
                    radii[j] -= diff * factor_j
                else:
                    radii[i] = 0
                    radii[j] = 0

    # Ensure non-negative
    radii = np.maximum(radii, 0)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    # Validate
    if not validate_packing(centers, radii):
        # If invalid, fallback to a safe grid
        # This should not happen if logic is correct
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.01)
        sum_radii = 0.26
        # Place in grid
        r = 0.04
        idx = 0
        y = r
        while idx < n:
            x = r
            while x + r <= 1 and idx < n:
                centers[idx] = [x, y]
                idx += 1
                x += 2*r
            y += 2*r
            if y + r > 1:
                break
        sum_radii = np.sum(radii)

    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Valid: {validate_packing(centers, radii)}")
