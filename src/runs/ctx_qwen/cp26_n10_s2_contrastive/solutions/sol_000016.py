# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abe07e0) state=2cecdc49 sum of radii=0.650000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed placement and iterative radius expansion strategy.
    """
    n = 26
    np.random.seed(42) # For reproducibility

    # 1. Initialization: Generate a hexagonal lattice pattern
    # We try to fit as many points as possible in a hex grid, then pick n.
    points = []
    # Approximate row spacing for hex grid: sqrt(3)/2 * diameter
    # But we don't know diameter yet. Let's generate a dense grid first.
    # Grid spacing 0.05
    for x in np.arange(0.05, 1.0, 0.05):
        for y in np.arange(0.05, 1.0, 0.05):
            points.append([x, y])
    
    # Shuffle and take first n
    np.random.shuffle(points)
    centers = np.array(points[:n])
    
    # Start with a small radius
    r = 0.01
    
    # 2. Optimization Loop
    # We will try to increase r step by step.
    # In each step, we resolve collisions.
    
    max_r = 0.0
    
    # Helper to check validity
    def check_valid(c, r_val):
        # Boundary
        if np.any(c[:, 0] < r_val - 1e-9) or np.any(c[:, 0] > 1 - r_val + 1e-9): return False
        if np.any(c[:, 1] < r_val - 1e-9) or np.any(c[:, 1] > 1 - r_val + 1e-9): return False
        # Overlaps
        dists_sq = np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=2)
        # Lower triangle
        mask = np.tril(np.ones((n, n)), k=-1).astype(bool)
        if np.any(dists_sq[mask] < (2 * r_val - 1e-9) ** 2):
            return False
        return True

    # Iterative expansion
    step_size = 0.0005
    max_iterations = 5000
    iteration = 0
    
    # Current radius
    current_r = 0.02
    
    while iteration < max_iterations:
        # Try to increase radius
        target_r = current_r + step_size
        
        # Run relaxation to fit target_r
        c_temp = centers.copy()
        resolved = True
        max_shift = 1e-5
        
        # Multiple passes to resolve overlaps
        for _ in range(20): # Local relaxation steps per radius increase
            shift_magnitude = 0.0
            
            # Check boundary constraints
            # Push back if too close to walls
            # Left wall
            mask_l = c_temp[:, 0] < target_r
            c_temp[mask_l, 0] = target_r
            # Right wall
            mask_r = c_temp[:, 0] > 1 - target_r
            c_temp[mask_r, 0] = 1 - target_r
            # Bottom wall
            mask_b = c_temp[:, 1] < target_r
            c_temp[mask_b, 1] = target_r
            # Top wall
            mask_t = c_temp[:, 1] > 1 - target_r
            c_temp[mask_t, 1] = 1 - target_r

            # Check pairwise overlaps
            # Vectorized computation of distances
            diff = c_temp[:, np.newaxis, :] - c_temp[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            dists = np.sqrt(np.maximum(dists_sq, 0))
            
            # Identify overlapping pairs
            # We need dist < 2*target_r
            # To save computation, only check lower triangle
            # Create a mask for i < j
            idx = np.triu_indices(n, k=1)
            row, col = idx
            
            pairs_dist = dists[row, col]
            pairs_diff = diff[row, col, :]
            
            overlap = pairs_dist < (2 * target_r)
            
            if np.any(overlap):
                # Resolve overlaps
                # For each overlapping pair, push apart
                # We need to update centers iteratively or batch update
                # Batch update might cause issues, so we do it carefully
                # Or just simple repulsion step
                
                forces = np.zeros_like(c_temp)
                
                for k in range(len(row)):
                    if overlap[k]:
                        i, j = row[k], col[k]
                        d = pairs_dist[k]
                        if d < 1e-9:
                            # Coincident, push random direction
                            vec = np.random.randn(2)
                            vec /= np.linalg.norm(vec)
                        else:
                            vec = -pairs_diff[k] / d
                        
                        # Displacement needed
                        push = (2 * target_r - d) / 2.0
                        
                        # Apply to forces
                        # Move i away from j, j away from i
                        forces[i] += vec * push
                        forces[j] -= vec * push
                
                # Apply forces (with damping to avoid oscillation)
                c_temp += forces * 0.5 # Damping factor
                shift_magnitude = np.max(np.abs(forces))
            
            if shift_magnitude < 1e-7:
                break
        
        # Check if configuration is valid
        if check_valid(c_temp, target_r):
            centers = c_temp
            current_r = target_r
            iteration += 1
            # If we made good progress, maybe increase step size?
            # But keep it small for precision
        else:
            # Failed to fit target_r
            # Try smaller step or stop?
            # Let's try smaller step to refine
            step_size /= 2.0
            if step_size < 1e-8:
                break
            # Don't increment iteration count if we just adjust step?
            # Or count it. Let's count it to avoid infinite loop on stuck state.
            iteration += 1

    # Final calculation
    final_r = current_r
    sum_radii = n * final_r
    
    return centers, np.full(n, final_r), sum_radii

# To test locally if run as script
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Radius: {radii[0]}")
    # Basic validation
    n = centers.shape[0]
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            print(f"Boundary violation: {i}")
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                print(f"Overlap: {i}, {j}")
    print(f"Valid: {valid}")
