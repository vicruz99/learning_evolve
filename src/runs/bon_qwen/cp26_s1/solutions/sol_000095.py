# sol_000095 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e7a6456) state=488e37bd sum of radii=1.585208 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    Uses initialization followed by iterative force-directed relaxation and expansion.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # 1. Initialize with a dense hexagonal grid
    # We place circles in a grid pattern that mimics a hexagonal lattice.
    # 26 circles can be arranged in roughly 5 rows.
    idx = 0
    # Try to cover the square with a 6x5 grid pattern, skipping some if needed
    # Hexagonal spacing factors: dx = 2r, dy = sqrt(3)r
    # We use a fixed spacing for initialization, r will grow later.
    
    # Let's just place them in a simple grid to start, the relaxation will fix it
    cols = 6
    rows = 5
    x_space = 1.0 / (cols + 1)
    y_space = 1.0 / (rows + 1)
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx < n:
                # Add slight jitter to break symmetry and avoid grid artifacts
                jitter = 0.02
                centers[idx, 0] = (c_idx + 1) * x_space + np.random.uniform(-jitter, jitter)
                centers[idx, 1] = (r_idx + 1) * y_space + np.random.uniform(-jitter, jitter)
                idx += 1
            else:
                break
        if idx >= n:
            break

    # 2. Simulation / Optimization Loop
    # We iterate to resolve overlaps and expand radii
    
    num_iterations = 2000
    # Adaptive step sizes
    dt = 0.01 
    expansion_rate = 1.0005
    
    for step in range(num_iterations):
        # Increase radii slightly to push for maximum sum
        # We do this every few steps or continuously
        radii *= (1 + 0.0001) 
        
        forces = np.zeros((n, 2))
        
        # Calculate repulsive forces between all pairs
        # O(N^2) is fine for N=26
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from j to i
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlap or too close. Apply repulsive force.
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        force_mag = overlap * 5.0 # Spring constant
                        forces[i] += (vec / dist) * force_mag
                        forces[j] -= (vec / dist) * force_mag
                    else:
                        # Circles on top of each other, random push
                        rand_vec = np.random.uniform(-1, 1, 2)
                        forces[i] += rand_vec
                        forces[j] -= rand_vec

        # Apply boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 5.0
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 5.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 5.0
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 5.0
        
        # Update positions
        # Limit maximum displacement per step to maintain stability
        max_displacement = 0.02
        forces = np.clip(forces, -max_displacement, max_displacement)
        centers += forces
        
        # Clamp centers to [0, 1]
        centers[:, 0] = np.clip(centers[:, 0], 0, 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, 1)

    # 3. Final Cleanup / Shrink to ensure validity
    # The simulation might leave tiny overlaps due to numerical precision.
    # We shrink radii slightly to guarantee validity with a margin.
    margin = 1e-4
    radii -= margin
    
    # If any radius became negative or too small, set to small positive
    radii = np.maximum(radii, 0.001)

    # Re-check validity and scale down if necessary
    # (A simple global scaling factor approach)
    valid = True
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            valid = False
            break
    
    if valid:
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j]:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Scale down radii uniformly to fit
        # Find the required scale factor
        # This is a conservative fallback
        scale = 1.0
        for _ in range(100):
            if check_packing_validity(centers, radii * scale):
                break
            scale *= 0.95
        radii *= scale

    return centers, radii, np.sum(radii)

def check_packing_validity(centers, radii):
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j]:
                return False
    return True
