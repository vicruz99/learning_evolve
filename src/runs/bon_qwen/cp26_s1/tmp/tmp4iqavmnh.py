import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def calculate_forces(centers, r):
    """
    Calculates repulsive forces for circles of radius r to resolve overlaps
    and boundary violations.
    Returns a force vector for each circle.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            min_dist = 2 * r
            
            if dist < min_dist and dist > 1e-9:
                # Repulsion force proportional to overlap
                # F = (min_dist - dist) * (diff / dist)
                # Using a simple linear repulsion
                overlap = min_dist - dist
                direction = diff / dist
                forces[i] += overlap * direction
                forces[j] -= overlap * direction
            elif dist <= 1e-9:
                # Prevent division by zero, push randomly
                rand_dir = np.random.randn(2)
                forces[i] += rand_dir
                forces[j] -= rand_dir

    # Boundary repulsion
    # Walls: x=0, x=1, y=0, y=1
    # If center x < r, push right. If x > 1-r, push left.
    for i in range(n):
        x, y = centers[i]
        
        # Left wall
        if x < r:
            forces[i, 0] += (r - x)
        # Right wall
        elif x > 1 - r:
            forces[i, 0] -= (x - (1 - r))
            
        # Bottom wall
        if y < r:
            forces[i, 1] += (r - y)
        # Top wall
        elif y > 1 - r:
            forces[i, 1] -= (y - (1 - r))
            
    return forces

def relax(centers, r, steps=1000, step_size=0.01):
    """
    Attempts to resolve overlaps and boundary violations for circles of radius r.
    Returns True if successful (no overlaps), False otherwise.
    """
    n = centers.shape[0]
    # Deep copy to avoid modifying original if we want, but here we modify in place
    current_centers = centers.copy()
    
    for _ in range(steps):
        forces = calculate_forces(current_centers, r)
        
        # Update positions
        current_centers += step_size * forces
        
        # Hard clamp to ensure they don't fly off, though forces should handle it
        # But strictly, center must be in [r, 1-r].
        # However, during relaxation, we might momentarily be outside if step is large?
        # Let's clamp strictly to [r, 1-r] to help convergence.
        current_centers = np.clip(current_centers, r, 1 - r)
        
        # Check if valid (quick check)
        # If max overlap is very small, we can stop early?
        # Let's just run full steps for simplicity and robustness
        
    return current_centers

def check_validity(centers, r):
    """
    Returns True if configuration with equal radii r is valid.
    """
    n = centers.shape[0]
    # Check boundary
    if np.any(centers < r - 1e-12) or np.any(centers > 1 - r + 1e-12):
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 2 * r - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    
    # 1. Initialization
    # Start with a hexagonal grid or random placement
    # Let's try a grid that fits roughly, then optimize
    # Hexagonal packing logic
    centers = np.zeros((N, 2))
    
    # Try to arrange in rows
    # 5 rows: 5, 5, 5, 5, 6? 
    # Let's just scatter them initially in the square with some spacing
    # Or use a deterministic grid
    # Grid 6x6 is 36 points. We need 26.
    # Let's take a subset of a 6x6 grid or just place them.
    
    # Better: Place in a 5x5 grid + 1
    # 5x5 grid spacing roughly 1/5 = 0.2
    # Let's start with random positions but ensure they are somewhat spread
    np.random.seed(42) # For reproducibility
    centers = np.random.rand(N, 2)
    
    # Initial radius small enough to be valid
    r = 0.02
    
    # Relax initial positions to spread them out a bit
    centers = relax(centers, r, steps=500, step_size=0.05)
    
    # 2. Iterative Expansion
    # We increase r and try to maintain validity
    r_current = 0.02
    r_best = r_current
    centers_best = centers.copy()
    
    # We want to maximize sum of radii. Assuming equal radii for now.
    # If we find a valid packing with radius r, sum is 26*r.
    
    # Step size for radius increase
    dr = 0.001
    
    # Loop to increase radius
    max_iterations = 2000
    iteration = 0
    
    # Current centers
    cur_centers = centers.copy()
    
    # We can use a while loop to increase r
    while iteration < max_iterations:
        r_try = r_current + dr
        
        # Relaxation to fit radius r_try
        # We need to run relaxation for the new larger radius
        # If relaxation fails to resolve overlaps, we backtrack
        
        # Save state
        prev_centers = cur_centers.copy()
        
        # Run relaxation with new radius
        # Increasing step size might help escape local minima but risks instability
        # Decreasing step size ensures stability
        new_centers = relax(cur_centers, r_try, steps=500, step_size=0.01)
        
        # Check validity
        if check_validity(new_centers, r_try):
            r_current = r_try
            cur_centers = new_centers
            r_best = r_current
            centers_best = cur_centers.copy()
            # Maybe decrease dr to fine tune? Or keep constant?
            # Let's keep constant or slowly decrease
            # dr *= 0.999 
            
            # Check if we hit a theoretical limit? 
            # Sum radii target is 2.636 => r ~ 0.1014
            if r_best > 0.105:
                 # We might be getting close to optimal, slow down?
                 pass
        else:
            # Failed to fit r_try.
            # Restore previous valid state
            r_current = r_best
            cur_centers = centers_best.copy()
            
            # Decrease dr to search more finely? 
            # Or just stop if dr is small enough
            if dr < 1e-5:
                break
            
            dr *= 0.9 # Reduce step size to find precise bound
            # Retry with smaller step? 
            # Actually, if we failed, we should just reduce dr and try again in next loop
            # But we are at the end of loop body.
            # Next iteration will try r_best + new_dr.
            
        iteration += 1
        
        # Safety break if r stops increasing
        if r_current <= r_best - 1e-6:
            # Something wrong, but r_current is tied to r_best logic
            pass

    # Final validation
    if not validate_packing(centers_best, np.full(N, r_best)):
        print("Final validation failed! Falling back to safe smaller radius.")
        # If it failed (due to numerical issues), reduce radius slightly
        while not validate_packing(centers_best, np.full(N, r_best)) and r_best > 0:
            r_best -= 0.001
            
    # Final check
    is_valid = validate_packing(centers_best, np.full(N, r_best))
    if not is_valid:
        print("Warning: Packing invalid after fallback.")
        
    sum_radii = np.sum(np.full(N, r_best))
    
    return centers_best, np.full(N, r_best), sum_radii

# Note: The problem statement implies we can return different radii.
# However, maximizing sum of radii often leads to equal radii for dense packings.
# If unequal radii were significantly better, a more complex optimizer would be needed.
# Given the constraints and time, the equal radius approach with good initialization
# and relaxation is the most robust strategy.