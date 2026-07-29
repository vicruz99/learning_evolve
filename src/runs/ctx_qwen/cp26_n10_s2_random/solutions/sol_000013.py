# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8a979775) state=151bad29 sum of radii=2.438240 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def repulsion_force(c1, r1, c2, r2, strength):
    dist_vec = c1 - c2
    dist = np.linalg.norm(dist_vec)
    min_dist = r1 + r2
    if dist < min_dist:
        overlap = min_dist - dist
        if dist > 1e-9:
            force_vec = (dist_vec / dist) * strength * overlap
        else:
            force_vec = np.random.rand(2) * strength
        return force_vec
    return np.zeros(2)

def boundary_force(c, r):
    force = np.zeros(2)
    if c[0] - r < 0:
        force[0] += 100.0 * (r - c[0])
    if c[0] + r > 1:
        force[0] -= 100.0 * (c[0] + r - 1)
    if c[1] - r < 0:
        force[1] += 100.0 * (r - c[1])
    if c[1] + r > 1:
        force[1] -= 100.0 * (c[1] + r - 1)
    return force

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)

    # 1. Generate initial configuration (perturbed hexagonal lattice)
    centers = []
    radii = []
    
    # Try to place circles in a hexagonal pattern
    r_est = 0.1
    d = 2 * r_est
    h = d * math.sqrt(3) / 2
    
    row = 0
    while True:
        y = r_est + row * h
        if y + r_est > 1.0:
            break
        
        # Determine x offset for the row
        offset = (row % 2) * (d / 2)
        x = r_est + offset
        
        while x + r_est <= 1.0:
            centers.append([x, y])
            radii.append(r_est)
            x += d
            if len(centers) >= n:
                break
        row += 1
        if len(centers) >= n:
            break

    # If we have fewer than 26, add some random ones in gaps
    if len(centers) < n:
        for _ in range(n - len(centers)):
            while True:
                c = np.random.rand(2)
                r = 0.05
                # Check if valid
                valid = True
                for i in range(len(centers)):
                    dist = np.linalg.norm(c - centers[i])
                    if dist < r + radii[i]:
                        valid = False
                        break
                if c[0] - r < 0 or c[0] + r > 1 or c[1] - r < 0 or c[1] + r > 1:
                    valid = False
                if valid:
                    centers.append(c)
                    radii.append(r)
                    break

    centers = np.array(centers[:n])
    radii = np.array(radii[:n])

    # 2. Optimization loop
    # We will iteratively optimize positions and radii
    # To maximize sum of radii, we want to expand them as much as possible
    
    # Let's run a few optimization steps with different parameters
    for opt_step in range(20):
        # Adjust repulsion strength and attraction
        repulsion_strength = 1000.0
        attraction_strength = 5.0
        boundary_strength = 500.0
        step_size = 0.001
        
        for _ in range(500):
            forces = np.zeros_like(centers)
            
            # Calculate repulsion between circles
            for i in range(n):
                for j in range(i + 1, n):
                    dist_vec = centers[i] - centers[j]
                    dist = np.linalg.norm(dist_vec)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 1e-6:
                            force_vec = (dist_vec / dist) * repulsion_strength * overlap
                        else:
                            force_vec = np.random.rand(2) * repulsion_strength
                        forces[i] += force_vec
                        forces[j] -= force_vec
            
            # Apply boundary forces
            for i in range(n):
                forces[i] += boundary_force(centers[i], radii[i])
                # Attraction to center to keep them from flying away, 
                # but we want to expand radii. 
                # A better approach: Push centers apart, and grow radii if not overlapping.
            
            centers += step_size * forces
            
            # Enforce boundaries strictly
            for i in range(n):
                centers[i][0] = max(radii[i], min(1.0 - radii[i], centers[i][0]))
                centers[i][1] = max(radii[i], min(1.0 - radii[i], centers[i][1]))

        # After positions are optimized, try to grow radii
        # Grow all radii uniformly until they hit a constraint
        min_gap = float('inf')
        for i in range(n):
            # Gap to walls
            min_gap = min(min_gap, centers[i][0] - radii[i])
            min_gap = min(min_gap, 1.0 - centers[i][0] - radii[i])
            min_gap = min(min_gap, centers[i][1] - radii[i])
            min_gap = min(min_gap, 1.0 - centers[i][1] - radii[i])
            
            # Gap to other circles
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    min_gap = min(min_gap, dist - radii[i] - radii[j])
        
        if min_gap > 1e-4:
            radii += min_gap
        else:
            # If stuck, try to increase radii slightly and re-optimize positions
            radii += 1e-4
            # Run a few more steps to resolve overlaps
            for _ in range(100):
                forces = np.zeros_like(centers)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist_vec = centers[i] - centers[j]
                        dist = np.linalg.norm(dist_vec)
                        min_dist = radii[i] + radii[j]
                        if dist < min_dist:
                            overlap = min_dist - dist
                            if dist > 1e-6:
                                force_vec = (dist_vec / dist) * repulsion_strength * overlap
                            else:
                                force_vec = np.random.rand(2) * repulsion_strength
                            forces[i] += force_vec
                            forces[j] -= force_vec
                centers += 0.001 * forces
                for i in range(n):
                    centers[i][0] = max(radii[i], min(1.0 - radii[i], centers[i][0]))
                    centers[i][1] = max(radii[i], min(1.0 - radii[i], centers[i][1]))

    # Final validation and correction
    # Ensure no overlaps and inside bounds
    for i in range(n):
        centers[i][0] = max(radii[i], min(1.0 - radii[i], centers[i][0]))
        centers[i][1] = max(radii[i], min(1.0 - radii[i], centers[i][1]))

    # One last check to shrink slightly if any overlap remains due to numerical errors
    for _ in range(10):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap_found = True
                    # Reduce radii to resolve
                    radii[i] *= 0.99
                    radii[j] *= 0.99
        if not overlap_found:
            break
            
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Final program wrapper
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Import numpy locally if not already imported in global scope for the function
    import numpy as np
    import math
    
    n = 26
    np.random.seed(42)

    # 1. Generate initial configuration (perturbed hexagonal lattice)
    centers = []
    radii = []
    
    # Try to place circles in a hexagonal pattern
    r_est = 0.1
    d = 2 * r_est
    h = d * math.sqrt(3) / 2
    
    row = 0
    while True:
        y = r_est + row * h
        if y + r_est > 1.0:
            break
        
        # Determine x offset for the row
        offset = (row % 2) * (d / 2)
        x = r_est + offset
        
        while x + r_est <= 1.0:
            centers.append([x, y])
            radii.append(r_est)
            x += d
            if len(centers) >= n:
                break
        row += 1
        if len(centers) >= n:
            break

    # If we have fewer than 26, add some random ones in gaps
    if len(centers) < n:
        for _ in range(n - len(centers)):
            while True:
                c = np.random.rand(2)
                r = 0.05
                # Check if valid
                valid = True
                for i in range(len(centers)):
                    dist = np.linalg.norm(c - centers[i])
                    if dist < r + radii[i]:
                        valid = False
                        break
                if c[0] - r < 0 or c[0] + r > 1 or c[1] - r < 0 or c[1] + r > 1:
                    valid = False
                if valid:
                    centers.append(c)
                    radii.append(r)
                    break

    centers = np.array(centers[:n])
    radii = np.array(radii[:n])

    # 2. Optimization loop
    repulsion_strength = 1000.0
    boundary_strength = 500.0
    
    # Multiple stages of optimization
    for opt_step in range(30):
        step_size = 0.005 * (0.95 ** opt_step)
        
        for _ in range(300):
            forces = np.zeros_like(centers)
            
            # Calculate repulsion between circles
            for i in range(n):
                for j in range(i + 1, n):
                    dist_vec = centers[i] - centers[j]
                    dist = np.linalg.norm(dist_vec)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 1e-6:
                            force_vec = (dist_vec / dist) * repulsion_strength * overlap
                        else:
                            force_vec = np.random.rand(2) * repulsion_strength
                        forces[i] += force_vec
                        forces[j] -= force_vec
            
            # Apply boundary forces
            for i in range(n):
                # Boundary push
                if centers[i][0] < radii[i]:
                    forces[i][0] += boundary_strength * (radii[i] - centers[i][0])
                elif centers[i][0] > 1.0 - radii[i]:
                    forces[i][0] -= boundary_strength * (centers[i][0] - (1.0 - radii[i]))
                
                if centers[i][1] < radii[i]:
                    forces[i][1] += boundary_strength * (radii[i] - centers[i][1])
                elif centers[i][1] > 1.0 - radii[i]:
                    forces[i][1] -= boundary_strength * (centers[i][1] - (1.0 - radii[i]))
                
                # Gentle attraction to center to prevent drifting too far if loose
                forces[i] -= 0.1 * (centers[i] - 0.5)
            
            centers += step_size * forces
            
            # Enforce boundaries strictly
            for i in range(n):
                centers[i][0] = max(radii[i], min(1.0 - radii[i], centers[i][0]))
                centers[i][1] = max(radii[i], min(1.0 - radii[i], centers[i][1]))

        # Try to grow radii uniformly
        min_gap = float('inf')
        for i in range(n):
            min_gap = min(min_gap, centers[i][0] - radii[i])
            min_gap = min(min_gap, 1.0 - centers[i][0] - radii[i])
            min_gap = min(min_gap, centers[i][1] - radii[i])
            min_gap = min(min_gap, 1.0 - centers[i][1] - radii[i])
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    min_gap = min(min_gap, dist - radii[i] - radii[j])
        
        if min_gap > 1e-5:
            radii += min_gap
        else:
            # If stuck, tiny increment and resolve
            radii += 1e-5

    # Final validation and correction
    for i in range(n):
        centers[i][0] = max(radii[i], min(1.0 - radii[i], centers[i][0]))
        centers[i][1] = max(radii[i], min(1.0 - radii[i], centers[i][1]))

    # Resolve any remaining overlaps by shrinking
    for _ in range(20):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap_found = True
                    # Reduce radii proportionally
                    radii[i] *= 0.99
                    radii[j] *= 0.99
        if not overlap_found:
            break
            
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Ensure the function is defined at top level as requested
# The prompt asks to return the final program between python tags.
# The function run_packing is defined.
