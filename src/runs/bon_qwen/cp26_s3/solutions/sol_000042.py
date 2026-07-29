# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9e7c8308) state=0549fb4b sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed layout with simulated annealing to find a high-density staggered arrangement.
    """
    np.random.seed(42)
    n_circles = 26
    
    # --- Phase 1: Initialization ---
    # Structure: 5, 6, 5, 6, 4 circles (Total 26)
    # This staggered layout is denser than a simple grid.
    row_configs = [5, 6, 5, 6, 4]
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    y_step = 0.18  # Vertical spacing between rows
    
    for row_idx, count in enumerate(row_configs):
        y_pos = 0.1 + row_idx * y_step
        
        # Shift every second row to create a staggered (hexagonal) effect
        x_offset = 0.1 if row_idx % 2 == 0 else 0.1 + 0.09
        
        # Space circles evenly within the row
        x_positions = np.linspace(x_offset, 1.0 - x_offset, count)
        
        for x in x_positions:
            centers[idx] = [x, y_pos]
            idx += 1

    # --- Phase 2: Force-Directed Optimization ---
    # Optimize positions to maximize minimum distance (and thus radius)
    iterations = 3000
    initial_temp = 0.1
    
    for step in range(iterations):
        # Simulated annealing: temperature decreases over time
        temp = initial_temp * (1.0 - step / iterations)
        
        forces = np.zeros_like(centers)
        
        # Calculate forces for each pair
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                    diff = np.random.rand(2) * 1e-4 # Break symmetry
                
                # Strong repulsive force to push circles apart
                force_mag = 0.5 / (dist ** 3)
                forces[i] += diff / dist * force_mag
                forces[j] -= diff / dist * force_mag
            
            # Boundary repulsion (push circles away from walls)
            for k in range(2):
                if centers[i][k] < 0.05:
                    forces[i][k] += 0.5 * (0.05 - centers[i][k])
                elif centers[i][k] > 0.95:
                    forces[i][k] -= 0.5 * (centers[i][k] - 0.95)

        # Update centers with forces and jitter
        jitter = np.random.randn(n_circles, 2) * temp * 0.01
        centers += forces * 0.05 + jitter
        
        # Hard constraint: Clip centers to valid range [0.001, 0.999]
        centers = np.clip(centers, 0.001, 0.999)

    # --- Phase 3: Calculate Radii ---
    # Determine the maximum uniform radius allowed by the optimized centers
    min_dist = 1.0
    
    # Check pairwise distances
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_dist:
                min_dist = dist
                
    # Check distance to boundaries
    for i in range(n_circles):
        d_to_boundary = min(centers[i][0], 1.0 - centers[i][0], 
                            centers[i][1], 1.0 - centers[i][1])
        if d_to_boundary * 2 < min_dist:
            min_dist = d_to_boundary * 2
            
    radius = min_dist / 2.0
    radii = np.full(n_circles, radius)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
