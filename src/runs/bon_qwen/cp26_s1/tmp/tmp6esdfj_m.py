import numpy as np

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float sum of radii
    """
    n = 26
    
    # --- 1. Initialization ---
    # Start with a hexagonal lattice arrangement to get a good initial geometry.
    # We will try to fit rows of circles. 
    # Approximate radius for equal circles is around 0.1.
    # Let's try to construct a layout with varying row lengths to fit 26.
    
    # Estimated radius for initialization
    r_init = 0.09
    
    centers = np.zeros((n, 2))
    radii = np.ones(n) * r_init
    
    # Construct hexagonal rows
    # Vertical spacing for hex packing: sqrt(3) * r
    # Horizontal spacing: 2 * r
    # Rows will alternate shift by r
    
    current_idx = 0
    row_idx = 0
    # Estimate number of rows needed. 
    # If we have ~5-6 circles per row, we need ~5 rows.
    # Let's try to fill rows greedily.
    
    y_pos = r_init
    max_y = 1.0 - r_init
    
    row_circles = []
    
    # Helper to add a circle
    def add_circle(x, y):
        nonlocal current_idx
        if current_idx < n:
            centers[current_idx, 0] = x
            centers[current_idx, 1] = y
            current_idx += 1

    # Try to fit circles in rows
    # Row 1: starts at x=r, spacing 2r
    # Row 2: starts at x=2r (shifted), spacing 2r
    # ...
    
    # We need to be careful not to exceed bounds.
    # Let's just generate a dense grid and pick first 26, or construct specifically.
    # A specific construction:
    # Row 0: 5 circles? Width 5*2r = 10r. If r=0.09, width 0.9. OK.
    # Row 1: 5 circles? Shifted.
    # ...
    
    # Let's just place them in a hexagonal pattern until we have 26.
    y = r_init
    row = 0
    while current_idx < n and y <= max_y:
        # Determine x start for this row
        # Even rows (0, 2, ...): start at r_init
        # Odd rows (1, 3, ...): start at 3*r_init (shifted by 2r? No, shift by r relative to neighbors)
        # Actually, standard hex:
        # Row 0: x = r, 3r, 5r... (centers at r, r+2r...)
        # Row 1: x = 2r, 4r, 6r... (centers at 2r, 2r+2r...) -> shift is r
        
        if row % 2 == 0:
            x_start = r_init
        else:
            x_start = 3 * r_init # shift by 2r? No.
            # If row 0 is at r, 3r. Distance between r and 3r is 2r.
            # Row 1 circles should be at distance 2r from row 0 circles.
            # Center of row 1 circle at x should satisfy (x - r)^2 + (sqrt(3)r)^2 = (2r)^2
            # (x-r)^2 = 4r^2 - 3r^2 = r^2 => x-r = r => x = 2r.
            # So shift is r. But row 0 starts at r. Row 1 should start at 2r.
            # Wait, if row 1 starts at 2r, is it inside? 
            # If r=0.1, 2r=0.2. OK.
            x_start = 2 * r_init # Shift by r relative to previous start? 
            # Actually, if row 0 is r, 3r, 5r.
            # Row 1 neighbors are at 2r, 4r.
            # Yes.
        
        x = x_start
        while current_idx < n and x <= 1.0 - r_init:
            add_circle(x, y)
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
        
    # If we didn't fill 26 (unlikely with r=0.09), or if we have gaps, 
    # the optimizer will handle it. But let's ensure we have 26.
    if current_idx < n:
        # Fallback: random placement for remaining
        for i in range(current_idx, n):
            centers[i, 0] = r_init + np.random.rand() * (1 - 2*r_init)
            centers[i, 1] = r_init + np.random.rand() * (1 - 2*r_init)
            
    # Reset radii to small value for optimization start
    radii[:] = 0.01
    
    # --- 2. Optimization (Repulsion Solver) ---
    
    velocities = np.zeros_like(centers)
    
    # Simulation parameters
    dt = 0.05
    damping = 0.9
    repulsion_strength = 5.0
    pressure = 0.0005 # Target pressure to grow radii
    max_iter = 2000
    
    # Precompute pair indices for efficiency
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    for step in range(max_iter):
        # 1. Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # We vectorize the calculation for pairs
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        
        ci = centers[i_idx]
        cj = centers[j_idx]
        
        diff = ci - cj # Vector from j to i
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(np.maximum(dist_sq, 1e-10))
        
        r_sum = radii[i_idx] + radii[j_idx]
        
        # Overlap amount
        overlap = r_sum - dist
        
        # Only push if overlapping
        overlap = np.maximum(overlap, 0)
        
        # Force direction
        if np.any(overlap > 0):
            # Force magnitude proportional to overlap
            # F = k * overlap
            # Direction is normalized diff
            # Avoid division by zero
            norm_factor = 1.0 / (dist + 1e-10)
            force_magnitude = repulsion_strength * overlap
            
            force_vec = diff * norm_factor[:, np.newaxis] * force_magnitude[:, np.newaxis]
            
            # Accumulate forces
            # i gets +force, j gets -force
            np.add.at(forces, i_idx, force_vec)
            np.add.at(forces, j_idx, -force_vec)
            
        # Boundary forces
        # Wall at 0: if x < r, push right
        # Wall at 1: if x + r > 1, push left
        
        # Left wall
        penetrate_left_x = np.maximum(radii - centers[:, 0], 0)
        forces[:, 0] += repulsion_strength * penetrate_left_x
        
        # Right wall
        penetrate_right_x = np.maximum((centers[:, 0] + radii) - 1.0, 0)
        forces[:, 0] -= repulsion_strength * penetrate_right_x
        
        # Bottom wall
        penetrate_bottom_y = np.maximum(radii - centers[:, 1], 0)
        forces[:, 1] += repulsion_strength * penetrate_bottom_y
        
        # Top wall
        penetrate_top_y = np.maximum((centers[:, 1] + radii) - 1.0, 0)
        forces[:, 1] -= repulsion_strength * penetrate_top_y
        
        # 2. Update velocities and positions
        velocities += forces * dt
        velocities *= damping
        centers += velocities * dt
        
        # 3. Grow radii
        # We try to increase radii slightly. 
        # If the system is stable (low forces/overlaps), we can grow faster.
        # Simple strategy: always increase by small amount, let forces push back if needed.
        radii += pressure
        
        # Optional: Check if radii are stuck or oscillating too much
        # But for this problem, steady growth is fine.
        
    # Final cleanup: ensure strict validity (clamping)
    # If any circle is slightly out due to numerical error, clamp it.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Clamp position
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        centers[i, 0] = x
        centers[i, 1] = y
        
    # Return results
    sum_radii = np.sum(radii)
    return centers, radii, float(sum_radii)

# To allow standalone execution for testing
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic check
    import math
    valid = True
    for i in range(26):
        if r[i] < 0: valid = False
        if c[i,0]-r[i] < -1e-9 or c[i,0]+r[i] > 1+1e-9: valid = False
        if c[i,1]-r[i] < -1e-9 or c[i,1]+r[i] > 1+1e-9: valid = False
    for i in range(26):
        for j in range(i+1, 26):
            d = math.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
            if d < r[i] + r[j] - 1e-9:
                valid = False
    print(f"Valid packing: {valid}")