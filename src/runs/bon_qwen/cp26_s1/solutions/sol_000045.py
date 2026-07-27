# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=bbe4ba84 sum of radii=2.026029 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (centers, radii, sum_radii) for 26 circles in a unit square.
    Uses a heuristic initialization followed by local optimization to maximize sum of radii.
    """
    n = 26
    np.random.seed(42)  # For reproducibility
    
    # 1. Initialize centers using a perturbed hexagonal grid
    centers = np.zeros((n, 2))
    idx = 0
    
    # Hexagonal packing parameters
    # We want to fit 26 circles. A rough estimate for equal circles is r ~ 0.1.
    # Row spacing ~ r * sqrt(3) ~ 0.173. Column spacing ~ 2r ~ 0.2.
    # We can fit about 5-6 rows.
    
    rows = []
    # Let's try to arrange them in rows of 5 and 6
    # 5, 5, 5, 5, 5, 1 -> 26? No.
    # 6, 5, 5, 5, 5 -> 26.
    row_counts = [6, 5, 6, 5, 4] # Total 26. 
    # Actually 6+5+6+5+4 = 26.
    # Let's verify if this fits.
    # Width for 6 circles: approx 1.0. Height for 5 rows: approx 1.0.
    
    current_y = 0.1
    row_idx = 0
    
    # Better initialization: random valid positions
    # But structured is better for optimization start.
    # Let's use a grid that is slightly compressed.
    
    # Simple grid init: 5x5 + 1
    # 25 circles in 5x5 grid, 1 in center of a gap?
    # Grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
    grid_coords = []
    for r in range(5):
        for c in range(5):
            grid_coords.append([0.1 + 0.2*r, 0.1 + 0.2*c])
    # Add one more in a gap, e.g., (0.2, 0.2)
    grid_coords.append([0.2, 0.2])
    
    # Shuffle to avoid symmetry bias during optimization
    centers = np.array(grid_coords)
    np.random.shuffle(centers)
    
    # Initial radii: small valid radius
    radii = np.full(n, 0.01)

    # 2. Optimization Loop
    # We will iteratively try to expand radii and move centers to relieve pressure.
    
    for iteration in range(200):
        # Calculate constraints for each circle
        # r_i <= min(
        #    x_i, 1-x_i, y_i, 1-y_i,
        #    min_j (dist(i,j) - r_j)
        # )
        # This is a system of inequalities. We can solve it by iteration (Jacobi method).
        
        # Copy current radii
        old_radii = radii.copy()
        
        for i in range(n):
            # Boundary constraints
            max_r = min(centers[i, 0], 1 - centers[i, 0], 
                        centers[i, 1], 1 - centers[i, 1])
            
            # Neighbor constraints
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # r_i + r_j <= dist  => r_i <= dist - r_j
                # We use the 'old' radii or current? 
                # Using current radii of neighbors might be too restrictive if they are shrinking.
                # But for expansion, we assume neighbors stay fixed for a moment.
                # Actually, let's use the radii from the start of this iteration for neighbors?
                # Or just use the current radii array.
                # To be safe and convergent, let's use old_radii for neighbors.
                constraint = dist - old_radii[j]
                if constraint < max_r:
                    max_r = constraint
            
            # Ensure non-negative
            max_r = max(0, max_r)
            radii[i] = max_r
            
        # Check convergence
        if np.max(np.abs(radii - old_radii)) < 1e-6:
            # Radii converged for fixed centers.
            # Now try to move centers to improve sum.
            pass
            
        # Perturb centers to escape local minima and increase sum
        # Simple gradient-free move: move circles away from their closest constraints
        if iteration % 2 == 0:
            for i in range(n):
                # Find closest neighbor or wall
                min_dist = float('inf')
                push_vec = np.array([0.0, 0.0])
                
                # Check walls
                walls = [
                    (centers[i, 0], -1, 0),      # Left wall, push right
                    (1 - centers[i, 0], 1, 0),   # Right wall, push left (vector -1) -> actually push away from wall
                    (centers[i, 0], 0, 0), # Wait, simpler:
                ]
                # Let's just check distance to walls and push away
                d_left = centers[i, 0]
                d_right = 1 - centers[i, 0]
                d_down = centers[i, 1]
                d_up = 1 - centers[i, 1]
                
                # We want to move away from the closest wall if it's constraining the radius
                # Radius is limited by min(d_left, d_right, d_down, d_up)
                # If r_i is close to min_dist, we should move away from that wall.
                
                min_wall_dist = min(d_left, d_right, d_down, d_up)
                if min_wall_dist <= radii[i] + 0.001: # Close to wall constraint
                    if d_left == min_wall_dist:
                        push_vec = np.array([0.1, 0]) # Move right
                    elif d_right == min_wall_dist:
                        push_vec = np.array([-0.1, 0]) # Move left
                    elif d_down == min_wall_dist:
                        push_vec = np.array([0, 0.1]) # Move up
                    elif d_up == min_wall_dist:
                        push_vec = np.array([0, -0.1]) # Move down
                        
                    # Apply small move
                    centers[i] += push_vec * 0.1 # Small step
                    # Clamp
                    centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])

                # Check neighbors
                for j in range(n):
                    if i == j: continue
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    required_dist = radii[i] + radii[j]
                    if dist < required_dist + 0.001: # Overlapping or touching
                        # Push i away from j
                        vec = centers[i] - centers[j]
                        if np.linalg.norm(vec) > 1e-9:
                            vec = vec / np.linalg.norm(vec)
                            centers[i] += vec * 0.05 # Small repulsion
                            # Clamp
                            centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])

    # Final validation and clipping
    # Re-calculate max possible radii one last time with fixed centers
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], 
                    centers[i, 1], 1 - centers[i, 1])
        for j in range(n):
            if i == j: continue
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            max_r = min(max_r, dist - radii[j])
        radii[i] = max(0, max_r)

    # Ensure centers are valid for the calculated radii
    # Sometimes optimization moves centers such that r > dist_to_wall
    # Clip radii to fit centers
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], 
                    centers[i, 1], 1 - centers[i, 1])
        radii[i] = min(radii[i], max_r)
        
    # One final check for overlaps and reduce radii if needed
    # This is a simple iterative reduction
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    # Overlap. Reduce radii proportionally or equally?
                    # To maximize sum, reduce the smaller one less? 
                    # Actually, reducing sum is bad. But we must be valid.
                    # Just split the deficit.
                    excess = radii[i] + radii[j] - dist
                    r_i_new = radii[i] - excess / 2
                    r_j_new = radii[j] - excess / 2
                    # Ensure non-negative
                    if r_i_new < 0: r_i_new = 0
                    if r_j_new < 0: r_j_new = 0
                    
                    radii[i] = r_i_new
                    radii[j] = r_j_new
                    changed = True

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Helper to run the function
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Centers shape: {c.shape}, Radii shape: {r.shape}")
    # Basic validation print
    # import validate_packing
    # print(validate_packing(c, r))
