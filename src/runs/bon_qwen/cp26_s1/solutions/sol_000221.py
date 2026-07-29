# sol_000221 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=a3a77849 sum of radii=1.950000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
    for i in range(n):
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    centers = []
    # Parameters for hex grid
    row_count = 6
    for i in range(row_count):
        # Y coordinate with vertical spacing
        y = 0.15 + i * 0.15 
        # Number of circles in this row
        num_in_row = 5 if i % 2 == 0 else 4
        # X coordinates
        for j in range(num_in_row):
            if len(centers) >= n:
                break
            x = 0.15 + j * 0.15 + (i % 2) * 0.075 # Stagger odd rows
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    
    # 2. Iterative Optimization (Repulsion/Inflation)
    radii = np.full(n, 0.01)
    
    # Parameters for the simulation
    learning_rate = 0.05
    repulsion_strength = 1.0
    damping = 0.9
    
    for step in range(2000):
        # Update radii based on current positions
        for i in range(n):
            # Distance to walls
            min_wall = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            # Distance to other circles
            min_dist = 1.0
            for j in range(n):
                if i == j:
                    continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    min_dist = dist
            
            # Radius is limited by walls and half the distance to nearest neighbor
            new_r = min(min_wall, min_dist / 2.0)
            radii[i] = new_r
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    # Target distance is sum of radii
                    target_dist = radii[i] + radii[j]
                    
                    if dist < target_dist:
                        # Overlap: Repulsion force
                        overlap = target_dist - dist
                        force_mag = (overlap * repulsion_strength) / dist
                        forces[i] += diff * force_mag
                        forces[j] -= diff * force_mag
            else:
                # Prevent division by zero
                pass

        # Apply boundary forces (push towards center if too close to wall)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 10.0
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 10.0
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10.0

        # Update positions
        centers += forces * learning_rate
        
        # Clip to square (safety)
        centers = np.clip(centers, 0, 1)

    # 3. Final Radius Calculation
    final_radii = np.zeros(n)
    for i in range(n):
        min_wall = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        min_dist = 1.0
        for j in range(n):
            if i == j: continue
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = min(min_dist, dist)
        
        final_radii[i] = min(min_wall, min_dist / 2.0)

    # 4. Final Adjustment to ensure strict non-overlap (epsilon margin)
    # We reduce radii slightly if necessary to handle float precision
    # But usually the min logic above ensures dist >= 2r.
    
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii

# Run and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(validate_packing(c, r))
