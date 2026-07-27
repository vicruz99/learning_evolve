# sol_000201 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a6b07ba) state=4ecdc9f4 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (centers, radii, sum_radii) for 26 circles in a unit square.
    Uses a force-directed optimization to maximize the minimum separation, 
    thereby maximizing the equal radii.
    """
    n = 26
    best_centers = None
    best_r = 0.0
    best_sum = 0.0

    def calculate_min_separation(centers):
        """Calculates the minimum distance between centers or boundaries."""
        # Pairwise distances
        dists = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.hypot(dx, dy)
                if d < dists:
                    dists = d
        
        # Boundary distances
        # x distance to 0 and 1
        dx = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        dy = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        min_boundary = np.minimum(dx.min(), dy.min())
        
        return np.minimum(dists, min_boundary)

    def optimize_packing(init_centers, steps=2000):
        """Runs force-directed optimization."""
        centers = init_centers.copy()
        # Learning rate parameters
        lr = 0.05
        cooling = 0.995
        
        for _ in range(steps):
            forces = np.zeros_like(centers)
            
            # Calculate repulsion forces
            for i in range(n):
                # Boundary repulsion
                for d in [centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]]:
                    if d < 0.3: # Only consider close boundaries
                        f = 1.0 / (d**2 + 1e-5)
                        # Direction
                        if d == centers[i, 0]: forces[i, 0] += f
                        elif d == 1 - centers[i, 0]: forces[i, 0] -= f
                        elif d == centers[i, 1]: forces[i, 1] += f
                        else: forces[i, 1] -= f

                # Circle repulsion
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    
                    if dist < 0.5: # Optimization cutoff
                        # Stronger repulsion when close
                        f_mag = 1.0 / (dist_sq + 1e-5)
                        forces[i, 0] += (dx / dist) * f_mag
                        forces[i, 1] += (dy / dist) * f_mag
                        forces[j, 0] -= (dx / dist) * f_mag
                        forces[j, 1] -= (dy / dist) * f_mag
            
            centers += forces * lr
            centers = np.clip(centers, 0.001, 0.999) # Keep inside strictly
            lr *= cooling

        return centers

    # --- Generation Strategies ---
    
    # Strategy 1: Hexagonal Grid
    # Rows of alternating counts. e.g., 5, 6, 5, 6, 4 (Total 26)
    h_centers = []
    rows = [5, 6, 5, 6, 4]
    row_y = 0.0
    spacing_y = 0.18
    spacing_x = 0.17
    shift = 0.085
    
    idx = 0
    for i, count in enumerate(rows):
        x_start = 0.5 - (count * spacing_x) / 2
        if i % 2 == 1:
            x_start += shift
        for _ in range(count):
            h_centers.append([x_start, row_y])
            x_start += spacing_x
        row_y += spacing_y
    
    if len(h_centers) == n:
        h_centers = np.array(h_centers)
        # Center and scale to fit roughly in [0.1, 0.9]
        h_centers = (h_centers - h_centers.min(axis=0)) / (h_centers.max(axis=0) - h_centers.min(axis=0)) * 0.8 + 0.1
        
        opt_centers = optimize_packing(h_centers)
        r = calculate_min_separation(opt_centers) / 2
        if r > best_r:
            best_r = r
            best_centers = opt_centers
            best_sum = n * r

    # Strategy 2: Perturbed 5x5 Grid + 1
    # Start with 25 in 5x5, place 26th in center, then optimize
    p_centers = []
    for i in range(5):
        for j in range(5):
            p_centers.append([0.1 + j*0.2, 0.1 + i*0.2])
    p_centers.append([0.5, 0.5]) # The 26th circle
    p_centers = np.array(p_centers)
    
    # Add small noise to break symmetry
    p_centers += np.random.uniform(-0.01, 0.01, p_centers.shape)
    p_centers = np.clip(p_centers, 0.01, 0.99)
    
    opt_centers = optimize_packing(p_centers, steps=3000)
    r = calculate_min_separation(opt_centers) / 2
    if r > best_r:
        best_r = r
        best_centers = opt_centers
        best_sum = n * r

    # Strategy 3: Random initialization (multiple restarts)
    for _ in range(5):
        r_centers = np.random.uniform(0.1, 0.9, (n, 2))
        opt_centers = optimize_packing(r_centers, steps=1000)
        r = calculate_min_separation(opt_centers) / 2
        if r > best_r:
            best_r = r
            best_centers = opt_centers
            best_sum = n * r

    # Finalize
    if best_centers is None:
        # Fallback to simple grid
        best_centers = np.array([[0.1 + j*0.2, 0.1 + i*0.2] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        best_r = 0.05 # Safe radius
        best_sum = n * best_r

    # Refine radius based on actual calculated separation
    final_r = calculate_min_separation(best_centers) / 2
    final_radii = np.full(n, final_r)
    final_sum = np.sum(final_radii)

    return best_centers, final_radii, final_sum

# Note: To ensure reproducibility and robustness in the returned solution,
# we rely on the optimization finding a local optimum near the theoretical max.
# The target 2.636 requires r ~ 0.1014. Hexagonal packing should approach this.
