import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    # Strategy:
    # 1. Define initial centers for a 5x5 grid plus a central "gap" circle.
    # 2. Run an iterative optimization to maximize sum of radii while respecting constraints.
    
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 1. Initialization
    # Create a 5x5 grid of circles
    grid_step = 0.2
    grid_offset = 0.1
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx, 0] = grid_offset + j * grid_step
            centers[idx, 1] = grid_offset + i * grid_step
            radii[idx] = 0.1
            idx += 1
    
    # Add the 26th circle in the center of the grid gap (0.5, 0.5)
    # We start with a small radius so it fits, optimizer will grow it.
    centers[25, 0] = 0.5
    centers[25, 1] = 0.5
    radii[25] = 0.01
    
    # 2. Optimization Loop
    # We use a simple iterative approach: 
    # For each circle, calculate the maximum possible radius allowed by neighbors/boundary,
    # then update. We repeat this to find a stable high-sum configuration.
    
    def get_max_radius(centers, radii, current_idx):
        """Calculate max radius for current_idx based on current centers and other radii."""
        x, y = centers[current_idx]
        max_r = min(x, 1-x, y, 1-y) # Boundary constraint
        
        for i in range(n):
            if i == current_idx:
                continue
            dist = np.sqrt((centers[current_idx, 0] - centers[i, 0])**2 + 
                           (centers[current_idx, 1] - centers[i, 1])**2)
            # Constraint: dist >= r_current + r_neighbor
            # r_current <= dist - r_neighbor
            r_constraint = dist - radii[i]
            if r_constraint < max_r:
                max_r = r_constraint
        return max_r

    # Iterative expansion
    # We run multiple passes to let the circles settle into a high-sum state
    for _ in range(500):
        # Randomize order to avoid bias
        order = np.random.permutation(n)
        for i in order:
            r_max = get_max_radius(centers, radii, i)
            # Apply a smoothing factor or direct update
            # To ensure convergence, we can slowly grow radii or set to max
            # Setting to max is aggressive but works for simple configurations
            # We add a tiny epsilon to prevent immediate re-shrinking if equal
            radii[i] = r_max + 1e-9 

    # 3. Final Adjustment and Validation
    # Ensure no NaNs or negative values
    radii = np.maximum(radii, 1e-6)
    
    # Final optimization pass to maximize sum explicitly
    # This is a greedy local search to push the sum higher
    # We try to expand the smallest circles first to fill gaps
    for _ in range(1000):
        # Find circle with largest gap (difference between current r and max possible r)
        # This helps equalize "pressure"
        max_gap_idx = -1
        max_gap = -1.0
        
        for i in range(n):
            r_max = get_max_radius(centers, radii, i)
            gap = r_max - radii[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        
        if max_gap_idx != -1 and max_gap > 1e-6:
            # Grow this circle slightly
            growth = max_gap * 0.1 
            radii[max_gap_idx] += growth
            
            # If growing causes overlap, we rely on the get_max_radius logic in next steps
            # But for a more robust packing, we could also perturb centers.
            # For this specific hybrid grid, center perturbation might be complex.
            # However, we can slightly nudge centers to relieve pressure.
            if get_max_radius(centers, radii, max_gap_idx) < radii[max_gap_idx] - 1e-7:
                # Overlap detected after growth, need to push centers apart
                # This is a simplification; a full optimizer would move all.
                pass 

    # Recalculate final valid radii to ensure validity
    # (In case the greedy growth pushed them over)
    valid_radii = np.zeros(n)
    for i in range(n):
        valid_radii[i] = get_max_radius(centers, radii, i)
    
    sum_radii = np.sum(valid_radii)
    
    return centers, valid_radii, sum_radii

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")