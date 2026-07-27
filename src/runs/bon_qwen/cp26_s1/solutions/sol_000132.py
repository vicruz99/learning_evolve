# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0b92a944) state=dfe70aca sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a force-directed expansion algorithm starting from a perturbed grid.
    """
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Perturbed grid
    # We initialize centers in a grid pattern with small random perturbations.
    # A 6x5 grid provides 30 points; we take the first 26.
    cols = 6
    rows = 5
    centers = []
    count = 0
    step_x = 1.0 / (cols + 1)
    step_y = 1.0 / (rows + 1)
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if count < n:
                x = (c_idx + 1) * step_x + np.random.uniform(-0.02, 0.02)
                y = (r_idx + 1) * step_y + np.random.uniform(-0.02, 0.02)
                centers.append([x, y])
                count += 1
    
    centers = np.array(centers)
    # Ensure initial centers are safely inside the square
    centers = np.clip(centers, 0.05, 0.95)
    
    # Initial radii
    radii = np.full(n, 0.01)
    
    # 2. Optimization
    # Iteratively expand radii and resolve overlaps using repulsive forces.
    iterations = 3000
    expansion_rate = 0.00015
    move_step = 0.02
    k_circle = 50.0  # Repulsion strength between circles
    k_wall = 100.0   # Repulsion strength from walls
    
    for it in range(iterations):
        # Increase radii to push for a better solution
        radii[:] += expansion_rate
        
        # Compute forces on each center
        force = np.zeros_like(centers)
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Boundary forces: push centers away if they violate boundaries
            if x < r: 
                force[i, 0] += (r - x) * k_wall
            if x > 1.0 - r: 
                force[i, 0] -= (x - (1.0 - r)) * k_wall
            if y < r: 
                force[i, 1] += (r - y) * k_wall
            if y > 1.0 - r: 
                force[i, 1] -= (y - (1.0 - r)) * k_wall
            
            # Inter-circle repulsion forces
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-12:
                        nx = dx / dist
                        ny = dy / dist
                        f = overlap * k_circle
                        force[i, 0] += nx * f
                        force[i, 1] += ny * f
                        force[j, 0] -= nx * f
                        force[j, 1] -= ny * f
                    else:
                        # If centers coincide, push randomly
                        force[i] += np.random.uniform(-1, 1, 2)
                        force[j] -= np.random.uniform(-1, 1, 2)
        
        # Move centers based on forces
        centers += force * move_step
        
        # Clamp centers to [0, 1] to maintain valid domain
        centers = np.clip(centers, 0.0, 1.0)
        
    # 3. Validation and Adjustment
    # Check if the resulting packing is strictly valid according to the validator's tolerance.
    # Validator uses a tolerance of 1e-12.
    tol = 1e-12
    valid = True
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            valid = False
            break
    
    # Check overlaps
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - tol:
                    valid = False
                    break
            if not valid:
                break
            
    # If invalid, scale down radii to ensure validity
    if not valid:
        low = 0.0
        high = 1.0
        # Binary search for the maximum scaling factor that yields a valid packing
        for _ in range(50):
            mid = (low + high) / 2.0
            test_radii = radii * mid
            ok = True
            
            # Check boundaries for scaled radii
            for i in range(n):
                x, y = centers[i]
                r = test_radii[i]
                if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
                    ok = False
                    break
            
            # Check overlaps for scaled radii
            if ok:
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                        if dist < test_radii[i] + test_radii[j] - tol:
                            ok = False
                            break
                    if not ok:
                        break
            
            if ok:
                low = mid
            else:
                high = mid
        
        radii = radii * low
        
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
