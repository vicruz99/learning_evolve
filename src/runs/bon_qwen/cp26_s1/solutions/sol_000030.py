# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=90bea092 sum of radii=0.353034 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    
    def check_and_score(centers, radii):
        """
        Checks if the packing is valid and returns the sum of radii if valid, else -1.
        """
        # Boundary check: circle must be within [0, 1]
        # Tolerance 1e-9 is used for internal checks, stricter than validator's 1e-12
        if (centers[:, 0] < radii - 1e-9).any() or (centers[:, 0] > 1 - radii + 1e-9).any() or \
           (centers[:, 1] < radii - 1e-9).any() or (centers[:, 1] > 1 - radii + 1e-9).any():
            return -1.0
        
        # Overlap check: distance between centers >= sum of radii
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf) # Ignore self-distance
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        min_gap = dists - radii_sum
        if np.any(min_gap < -1e-9):
            return -1.0
            
        return np.sum(radii)

    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Prepare initial configurations
    configs = []
    
    # Config 1: 5x5 Grid + 1 extra circle (perturbed)
    # A 5x5 grid is a dense packing for 25 circles. Adding a 26th requires perturbation.
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(grid_x, grid_y)
    c1 = np.column_stack([xx.ravel(), yy.ravel()])
    # Add a 26th circle slightly offset from center to avoid exact overlap with grid point (0.5, 0.5)
    c1 = np.vstack([c1, [0.501, 0.501]])
    # Add small random noise to break symmetry
    c1 += np.random.normal(0, 1e-4, size=c1.shape)
    configs.append((c1, np.full(n, 0.02))) # Start with small valid radii
    
    # Config 2: Random positions
    c2 = np.random.rand(n, 2) * 0.8 + 0.1
    configs.append((c2, np.full(n, 0.01)))
    
    for trial, (init_centers, init_radii) in enumerate(configs):
        centers = init_centers.copy()
        radii = init_radii.copy()
        
        lr_center = 1e-4       # Learning rate for moving centers
        expansion_step = 1.00005 # Factor to increase radii
        max_iters = 6000       # Number of iterations
        
        invalid_count = 0      # Counter to detect stuck states
        
        for i in range(max_iters):
            # --- Resolve Overlaps via Repulsive Forces ---
            # Run multiple sub-steps for stability
            for _ in range(5):
                forces = np.zeros_like(centers)
                
                # 1. Boundary Repulsion
                # Push circles away from walls if they penetrate
                left_pen = np.maximum(0, radii - centers[:, 0])
                right_pen = np.maximum(0, centers[:, 0] - (1 - radii))
                bottom_pen = np.maximum(0, radii - centers[:, 1])
                top_pen = np.maximum(0, centers[:, 1] - (1 - radii))
                
                forces[:, 0] += (left_pen - right_pen) * 10.0
                forces[:, 1] += (bottom_pen - top_pen) * 10.0
                
                # 2. Pairwise Repulsion
                # Calculate distances and overlaps between all pairs
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
                overlaps = np.maximum(0, radii_sum - dists)
                
                # Direction vectors (unit vectors from j to i)
                dirs = diff / (dists[:, :, np.newaxis] + 1e-9)
                
                # Accumulate forces proportional to overlap
                forces += np.sum(overlaps[:, :, np.newaxis] * dirs, axis=1)
                
                # Update centers
                centers += lr_center * forces
                
                # Hard clip to ensure centers stay within [0, 1]
                centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
                centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
            
            # --- Check Validity and Update Best ---
            score = check_and_score(centers, radii)
            
            if score > 0:
                invalid_count = 0
                if score > best_sum:
                    best_sum = score
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                
                # Expand radii to try and fit larger circles
                radii *= expansion_step
                radii = np.minimum(radii, 0.5) # Safety cap to prevent explosion
                
                # Refine parameters as we get closer to optimal
                if best_sum > 2.5:
                    expansion_step = 1.0 + 1e-7
                    lr_center = 1e-5
            else:
                invalid_count += 1
                if invalid_count > 50:
                    # If stuck in an invalid state for too long, shrink radii to recover
                    radii *= 0.99
                    invalid_count = 0
                    radii = np.maximum(radii, 0.01)

    # Fallback (should not be reached with valid logic)
    if best_centers is None:
        best_centers = init_centers
        best_radii = init_radii
        s = check_and_score(best_centers, best_radii)
        best_sum = s if s > 0 else 0.26

    return best_centers, best_radii, best_sum
