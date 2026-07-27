import numpy as np

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a force-directed simulation with gradual circle growth.
    """
    n = 26
    
    # Configuration parameters
    iterations = 20000
    growth_rate = 0.00005  # How much radius increases per step
    k_repulsion = 5.0      # Force constant for overlap repulsion
    k_wall = 10.0          # Force constant for boundary constraints
    dt = 0.05              # Time step for position updates
    damping = 0.9          # Velocity damping to stabilize
    
    # 1. Initialization
    # Initialize centers in a hexagonal-like pattern to help packing
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.02) # Start with small radii
    
    # Hexagonal packing initialization
    # We try to fit them in a grid that fits within [0,1]
    # Estimate rows and cols
    # For 26 circles, maybe 6x5 or similar
    # Let's just scatter them randomly but with some spacing to avoid immediate huge overlaps
    # Random placement with rejection
    attempts = 0
    idx = 0
    while idx < n and attempts < 10000:
        attempts += 1
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        
        # Check distance to existing circles (using initial radius 0.02)
        overlap = False
        for i in range(idx):
            dist = np.sqrt((centers[i,0]-x)**2 + (centers[i,1]-y)**2)
            if dist < 0.05: # Keep some buffer
                overlap = True
                break
        if not overlap:
            centers[idx] = [x, y]
            idx += 1
            
    # If we couldn't place all, fill remaining randomly
    if idx < n:
        for i in range(idx, n):
            centers[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]

    # Initialize velocities
    velocities = np.zeros((n, 2))

    # 2. Simulation Loop
    for step in range(iterations):
        # Increase radii slowly
        # We can increase them more if the system is not too crowded, 
        # but constant slow growth is safer for convergence
        radii += growth_rate
        
        # Calculate forces
        forces = np.zeros((n, 2))
        
        # Pairwise interactions
        # Vectorized distance calculation for performance
        # Compute distance matrix
        # centers shape (n, 2)
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Set diagonal to infinity to ignore self-interaction
        np.fill_diagonal(dists, np.inf)
        
        # Find overlapping pairs
        # sum_radii matrix (n, n)
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: positive if overlapping
        overlap_amount = sum_radii - dists
        
        # Mask for overlaps
        is_overlapping = overlap_amount > 0
        
        # Calculate repulsive forces for overlapping pairs
        # Force vector direction is along the line connecting centers
        # F = k * overlap * unit_vector
        
        # To avoid division by zero for identical points (though unlikely with dists>0 check)
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        unit_vectors = diff / safe_dists[:, :, np.newaxis]
        
        # Force magnitude
        force_magnitude = k_repulsion * np.where(is_overlapping, overlap_amount, 0.0)
        
        # Apply forces
        # forces[i] gets contribution from all j
        # For each pair (i, j), force on i is -F * u_ij, force on j is +F * u_ij
        # We can sum this up.
        # force_field[i] = sum_j ( force_magnitude[i,j] * unit_vectors[i,j] ) ? 
        # Wait, unit_vectors[i,j] points from j to i? 
        # diff[i,j] = c_i - c_j. So unit_vectors[i,j] points from j to i.
        # If i and j overlap, we want to push i away from j (direction i-j) and j away from i (direction j-i).
        # So force on i should be proportional to (c_i - c_j).
        # Yes, unit_vectors[i,j] is correct direction for repulsion on i.
        
        # Force on circle i from circle j
        # F_ij = k * (r_i + r_j - d_ij) * (c_i - c_j) / d_ij
        # This is what we have: force_magnitude * unit_vectors
        
        # Sum forces on each circle
        # We need to be careful: the matrix force_magnitude is symmetric?
        # Yes, overlap_amount is symmetric.
        # But we only want to apply force if overlapping.
        
        # Accumulate forces
        # force_field[i] = sum_j ( force_magnitude[i,j] * unit_vectors[i,j] )
        # Note: unit_vectors[i,j] = - unit_vectors[j,i]
        # So this correctly applies equal and opposite forces.
        
        force_field = np.sum(force_magnitude[:, :, np.newaxis] * unit_vectors, axis=1)
        
        forces += force_field

        # Boundary forces
        # Push circles inside [0, 1] x [0, 1] respecting radius
        # Constraint: r <= x <= 1-r  =>  x-r >= 0 and 1-r-x >= 0
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall (x < r)
            if x < r:
                penetration = r - x
                forces[i, 0] += k_wall * penetration
            
            # Right wall (x > 1-r)
            elif x > 1 - r:
                penetration = x - (1 - r)
                forces[i, 0] -= k_wall * penetration
                
            # Bottom wall (y < r)
            if y < r:
                penetration = r - y
                forces[i, 1] += k_wall * penetration
                
            # Top wall (y > 1-r)
            elif y > 1 - r:
                penetration = y - (1 - r)
                forces[i, 1] -= k_wall * penetration

        # Update velocities and positions
        # a = F (assuming mass=1)
        # v = v + a * dt
        # x = x + v * dt
        
        velocities = velocities * damping + forces * dt
        centers = centers + velocities * dt

        # Clamp centers to stay roughly inside to prevent explosion, 
        # though wall forces should handle it. 
        # Hard clamp to [0, 1] just in case numerical errors push them out too far.
        # But we must respect radius for valid output. 
        # The optimizer should keep them in [r, 1-r]. 
        # If they are stuck, hard clamping might help recovery.
        # However, hard clamping x < 0 to 0 might cause violation if r > 0.
        # Let's trust the forces.

    # 3. Final Cleanup and Validation
    # Ensure radii are valid and positions are within bounds.
    # If a circle is slightly out, push it in.
    for i in range(n):
        r = radii[i]
        # Clamp x
        if centers[i, 0] < r: centers[i, 0] = r
        if centers[i, 0] > 1 - r: centers[i, 0] = 1 - r
        # Clamp y
        if centers[i, 1] < r: centers[i, 1] = r
        if centers[i, 1] > 1 - r: centers[i, 1] = 1 - r

    # Calculate sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii

# Helper to run and check (for local testing, though not required by prompt format)
if __name__ == "__main__":
    import numpy as np # Already imported
    
    # Run the packing
    centers, radii, sum_r = run_packing()
    
    # Basic validation
    print(f"Sum of radii: {sum_r}")
    print(f"Number of circles: {len(radii)}")
    
    # Check overlaps
    n = len(centers)
    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-6:
                overlap_count += 1
    print(f"Overlaps: {overlap_count}")
    
    # Check bounds
    out_of_bounds = 0
    for i in range(n):
        if (centers[i,0] < radii[i] or centers[i,0] > 1-radii[i] or
            centers[i,1] < radii[i] or centers[i,1] > 1-radii[i]):
            out_of_bounds += 1
    print(f"Out of bounds: {out_of_bounds}")